"""
03_sweep_parametros.py
======================
Estudo dos parametros da sintese: K (vizinhos), alpha0 (concentracao Dirichlet)
e SPAN_RATIO (faixa de extensao S aceita para os vizinhos).

Para cada rota (sentido de linha) sorteia ancoras em todas as faixas horarias.
Para cada faixa de SPAN, o alinhamento com as pernas candidatas e calculado uma
vez e reutilizado por toda a grade (K, alpha0).

Etapa A (fidelidade e variabilidade), pooled sobre rotas e ancoras:
  KS_T, KS_S, KS_v   distancia de Kolmogorov-Smirnov entre sintetico e real
                     para duracao T, extensao S e velocidade media S/T
  div_T, div_S       desvio-padrao sintetico / real (1 = mesma variabilidade)

Privacidade (para todas as configuracoes):
  razao_privacidade = mediana da distancia (curva gamma) da sintetica a perna
  real mais proxima / mediana da mesma distancia entre pernas reais
  (controle). Valor < 1 indica sinteticas mais proximas de reais do que reais
  entre si, ou seja, copia.

Criterio de escolha (definido antes de ver a grade final, ver ANALISE_RESULTADOS):
  - N_eff >= 2: a sintetica deve misturar pelo menos 2 vizinhos efetivos
    (N_eff = (alpha0 + 1)/(alpha0/K + 1), valor esperado de 1/sum(p^2));
  - div_T >= 0,75: a duracao nao pode encolher alem de 25 % da variabilidade real;
  - entre as configuracoes que cumprem as duas, minimiza-se KS_medio.
A razao de privacidade e reportada, mas nao e restricao (dados de onibus, nao
pessoais). div_S fica abaixo de 1 em toda a grade (limite estrutural da media
convexa: a cauda de pernas parciais nao entra nos vizinhos). Saidas em resultados/.

Uso com a grade ja calculada (sem refazer a etapa A):  --reusar

Uso:
    cd sintetic/pipeline && ../.venv/bin/python 03_sweep_parametros.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import itertools
import json
import multiprocessing as mp
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

import config as cfg
from core.shapes import load_shape_models
from core.trips import build_configs
from functional_synthesis_dirichlet import (build_neighbour_cache,
                                            synthesize_from_cache)
from legs import build_route_legs, HOUR_BUCKETS

warnings.filterwarnings("ignore")

RESULTS = cfg.BASE / "resultados"
K_GRID = [2, 4, 6, 8, 12]
ALPHA_GRID = [0.25, 0.5, 1.0, 2.0, 7.0]
SPAN_GRID = [1.05, 1.1, 1.25, 1.5, 2.0]
ANCHORS_PER_BUCKET = 5
DRAWS_PER_ANCHOR = 8
SEED = 7

_ROUTES = {}   # tag -> dict(db, meta)


def _anchor_task(args):
    """Etapa A: alinhamentos por SPAN; metricas por (K, alpha0): T, S e a
    distancia (curva gamma) de cada sintetica a perna real mais proxima."""
    tag, anchor_idx, bucket, seed = args
    db = _ROUTES[tag]["db"]
    G = _ROUTES[tag]["G"]
    rng = np.random.default_rng(seed)
    anchor = db[db["hour_bucket"] == bucket].iloc[anchor_idx]
    L = _ROUTES[tag]["meta"]["route_length_m"] if _ROUTES[tag]["meta"]["is_loop"] else None
    out = {}
    for span in SPAN_GRID:
        cache = build_neighbour_cache(db, anchor, span)
        if cache is None:
            continue
        for K, a0 in itertools.product(K_GRID, ALPHA_GRID):
            Ts, Ss, dmin = [], [], []
            for _ in range(DRAWS_PER_ANCHOR):
                r = synthesize_from_cache(cache, K, a0, rng, loop_length=L)
                Ts.append(r.T)
                Ss.append(r.S)
                dmin.append(np.sqrt(((G - r.gamma[None, :]) ** 2).sum(axis=1)).min())
            out[(span, K, a0)] = (np.array(Ts), np.array(Ss), np.array(dmin))
    return tag, out


def _real_control(tag):
    """Distancia de cada perna real a sua vizinha real mais proxima (controle)."""
    db = _ROUTES[tag]["db"]
    G = np.vstack(db["gamma"].to_numpy())
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(G), size=min(300, len(G)), replace=False)
    nn = []
    for i in idx:
        d = np.sqrt(((G - G[i][None, :]) ** 2).sum(axis=1))
        d[i] = np.inf
        nn.append(d.min())
    return np.array(nn)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reusar", action="store_true",
                    help="le resultados/sweep_resumo.csv em vez de recalcular")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.reusar:
        agg = pd.read_csv(RESULTS / "sweep_resumo.csv")
        return escolher(agg)
    shape_models = load_shape_models(cfg.SHAPE_PATH)
    configs = build_configs(shape_models, cfg.LINHAS_CIRCULARES, cfg.LINHAS_IDA_VOLTA)

    tasks, real = [], {}
    for (linha, sentido, _sid, _s, _e) in configs:
        r = build_route_legs(linha, sentido, shape_models, configs)
        tag = f"{linha}_{sentido}"
        G = np.vstack(r["db"]["gamma"].to_numpy())
        _ROUTES[tag] = dict(db=r["db"], meta=r["meta"], G=G)
        real[tag] = dict(T=r["db"]["T"].to_numpy(float), S=r["db"]["S"].to_numpy(float))
        rng = np.random.default_rng(SEED)
        for bucket in HOUR_BUCKETS:
            n_b = int((r["db"]["hour_bucket"] == bucket).sum())
            if n_b < 3:
                continue
            for _ in range(ANCHORS_PER_BUCKET):
                tasks.append((tag, int(rng.integers(n_b)), bucket,
                              int(rng.integers(1_000_000_000))))
        print(f"{tag}: {len(r['db'])} pernas", flush=True)

    try:
        mp.set_start_method("fork", force=True)
    except (RuntimeError, ValueError):
        pass

    print(f"\nEtapa A: {len(tasks)} ancoras x {len(SPAN_GRID)} spans x "
          f"{len(K_GRID) * len(ALPHA_GRID)} configuracoes (K, alpha0)", flush=True)
    with mp.Pool(processes=min(8, mp.cpu_count())) as pool:
        resA = pool.map(_anchor_task, tasks, chunksize=1)

    rows = []
    keys = [(s, K, a) for s in SPAN_GRID for K in K_GRID for a in ALPHA_GRID]
    for tag in real:
        Tr, Sr = real[tag]["T"], real[tag]["S"]
        vr = Sr / Tr
        for (span, K, a0) in keys:
            Ts = np.concatenate([o[(span, K, a0)][0] for t, o in resA
                                 if t == tag and (span, K, a0) in o])
            Ss = np.concatenate([o[(span, K, a0)][1] for t, o in resA
                                 if t == tag and (span, K, a0) in o])
            rows.append(dict(
                rota=tag, span=span, K=K, alpha0=a0,
                KS_T=ks_2samp(Ts, Tr).statistic,
                KS_S=ks_2samp(Ss, Sr).statistic,
                KS_v=ks_2samp(Ss / Ts, vr).statistic,
                div_T=np.std(Ts) / np.std(Tr),
                div_S=np.std(Ss) / np.std(Sr),
                n_sinteticos=len(Ts),
            ))
    df = pd.DataFrame(rows)
    df["KS_medio"] = df[["KS_T", "KS_S", "KS_v"]].mean(axis=1)
    df.to_csv(RESULTS / "sweep_K_alpha0.csv", index=False, float_format="%.5f")

    agg = (df.groupby(["span", "K", "alpha0"])
           .agg(KS_medio=("KS_medio", "mean"), KS_T=("KS_T", "mean"),
                KS_S=("KS_S", "mean"), KS_v=("KS_v", "mean"),
                div_T_med=("div_T", "median"), div_S_med=("div_S", "median"))
           .reset_index())
    agg["ok_div"] = (agg["div_T_med"] >= 0.8) & (agg["div_S_med"] >= 0.8)

    # privacidade: distancia das sinteticas a pernas reais vs. controle real-real
    controls = {tag: _real_control(tag) for tag in real}
    priv = []
    for tag in real:
        for (span, K, a0) in keys:
            dm = np.concatenate([o[(span, K, a0)][2] for t, o in resA
                                 if t == tag and (span, K, a0) in o])
            priv.append(dict(rota=tag, span=span, K=K, alpha0=a0,
                             razao_privacidade=np.median(dm) / np.median(controls[tag])))
    priv = pd.DataFrame(priv).groupby(["span", "K", "alpha0"], as_index=False)[
        "razao_privacidade"].median()
    agg = agg.merge(priv, on=["span", "K", "alpha0"], how="left")
    agg.to_csv(RESULTS / "sweep_resumo.csv", index=False, float_format="%.5f")

    return escolher(agg)


def escolher(agg: pd.DataFrame):
    """Aplica o criterio (N_eff >= 2, div_T >= 0,75, min KS_medio) e grava a escolha."""
    agg = agg.copy()
    agg["N_eff"] = (agg["alpha0"] + 1) / (agg["alpha0"] / agg["K"] + 1)
    agg["ok_criterio"] = (agg["div_T_med"] >= 0.75) & (agg["N_eff"] >= 2.0)
    agg.to_csv(RESULTS / "sweep_resumo.csv", index=False, float_format="%.5f")
    ok = agg[agg["ok_criterio"]]
    print(f"\nConfiguracoes que atendem o criterio: {len(ok)} de {len(agg)}")
    if len(ok) == 0:
        raise SystemExit("nenhuma configuracao atende o criterio; ver resultados/sweep_resumo.csv")
    best = ok.sort_values("KS_medio").iloc[0]
    escolha = dict(span_ratio=float(best["span"]), K=int(best["K"]),
                   alpha0=float(best["alpha0"]), N_eff=float(best["N_eff"]),
                   KS_medio=float(best["KS_medio"]),
                   div_T_med=float(best["div_T_med"]), div_S_med=float(best["div_S_med"]),
                   razao_privacidade=float(best["razao_privacidade"]),
                   criterio="N_eff >= 2 e div_T >= 0,75; minimiza KS_medio")
    (RESULTS / "escolha_parametros.json").write_text(json.dumps(escolha, indent=2))

    print("\nTop 10 por KS_medio entre as que atendem o criterio:")
    cols = ["span", "K", "alpha0", "N_eff", "KS_medio", "KS_T", "KS_S", "KS_v",
            "div_T_med", "div_S_med", "razao_privacidade"]
    print(ok.sort_values("KS_medio")[cols].head(10).round(3).to_string(index=False))
    ref = agg[(agg.span == 2.0) & (agg.K == 8) & (agg.alpha0 == 7.0)]
    print("\nReferencia anterior (span=2.0, K=8, alpha0=7):")
    print(ref[cols].round(3).to_string(index=False))
    print(f"\nEscolha: {escolha}")


if __name__ == "__main__":
    main()
