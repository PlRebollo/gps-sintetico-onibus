"""
04_gerar_sinteticos.py
======================
PASSO 4: gera as viagens sinteticas de todas as rotas (linha x sentido) com a
sintese Dirichlet corrigida (functional_synthesis_dirichlet.py) e os parametros
escolhidos no estudo 03 (config.py: SYNTH_K, SYNTH_ALPHA0, SPAN_RATIO).

As faixas sao as da rota (meta["faixas"], ver faixas.py). Cada rota recebe --n sinteticas, repartidas
entre as faixas na proporcao das pernas reais (--alocacao proporcional, padrao) ou
N_PER_BUCKET por faixa (--alocacao fixa, como no pipeline original). Hora de inicio
a partir das partidas reais da faixa; dia de marco/2026 sorteado entre os dias do
tipo da faixa (util ou sabado/domingo).

Saidas (dados_sinteticos/):
  synthetic_fulltrip_dirichlet_{LINHA}.csv.gz   pontos a 1 Hz de todas as viagens
      colunas: linha, sentido, hour_bucket, synth_leg_id, anchor_leg_id,
               t_s, dt, lat, lon, s_m, v_ms, lateral_residual_m
  metadados_sinteticos.csv                      uma linha por viagem sintetica

Uso:
    cd pipeline && python 04_gerar_sinteticos.py
        [--procs N] [--n 150] [--alocacao proporcional|fixa] [--sem_pontos] [--saida DIR]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import multiprocessing as mp
import warnings

import numpy as np
import pandas as pd

import config as cfg
from core.shapes import load_shape_models
from core.trips import build_configs
from functional_synthesis_dirichlet import (build_neighbour_cache, bucket_start_hour,
                                            result_to_frame, sample_anchor,
                                            synthesize_from_cache)
from legs import build_route_legs, load_route

warnings.filterwarnings("ignore")

MARCH_START = pd.Timestamp("2026-03-01")
DIAS = pd.date_range(MARCH_START, periods=cfg.MARCH_DAYS)

_ROUTES = {}   # tag -> dict com banco, forma da rota, L, sigma, linha, sentido


def _gerar(args):
    tag, bucket, i, seed = args
    R = _ROUTES[tag]
    rng = np.random.default_rng(seed)
    cache = None
    for _ in range(20):                       # ancora sem vizinhos suficientes: tenta outra
        anchor = sample_anchor(R["db"], bucket, rng)
        cache = build_neighbour_cache(R["db"], anchor, cfg.SPAN_RATIO)
        if cache is not None and len(cache.ids) >= cfg.SYNTH_K:
            break
    if cache is None:
        return None
    res = synthesize_from_cache(cache, cfg.SYNTH_K, cfg.SYNTH_ALPHA0, rng,
                                loop_length=R["L"] if R["is_loop"] else None)
    fds = R["faixas"][bucket][2]
    dias = DIAS if fds is None else DIAS[(DIAS.dayofweek >= 5) == fds]
    day = dias[int(rng.integers(len(dias)))]
    t0 = (day + pd.Timedelta(hours=bucket_start_hour(bucket, rng, R["faixas"],
                                                     R["horas"].get(bucket)))).floor("s")
    synth_id = f"{tag}_{bucket}_{i:03d}"
    frame = None
    if not _SEM_PONTOS:
        frame = result_to_frame(res, R["shape"], R["sigma"], t0,
                                R["L"] if R["is_loop"] else None, rng)
        frame.insert(0, "anchor_leg_id", res.anchor_id)
        frame.insert(0, "synth_leg_id", synth_id)
        frame.insert(0, "hour_bucket", bucket)
        frame.insert(0, "sentido", R["sentido"])
        frame.insert(0, "linha", R["linha"])

    w = res.weights
    meta = dict(
        linha=R["linha"], sentido=R["sentido"], hour_bucket=bucket,
        synth_leg_id=synth_id, anchor_leg_id=res.anchor_id,
        inicio=t0, duracao_s=res.T, extensao_m=res.S, s0_m=res.s0,
        velocidade_media_kmh=3.6 * res.S / res.T,
        n_candidatos=res.n_candidates, n_vizinhos=len(res.neighbour_ids),
        N_eff=float(1.0 / np.sum(w ** 2)), peso_max=float(w.max()),
        D_medio_vizinhos=float(np.mean(res.D)),
        frac_tempo_teto_velocidade=res.clip_frac,
        v_p99_1hz_kmh=3.6 * float(np.percentile(res.v, 99)),
    )
    return frame, meta


_SEM_PONTOS = False


def alocar(contagens: dict, n_total: int, alocacao: str) -> dict:
    """Sinteticas por faixa: N_PER_BUCKET cada (fixa) ou proporcional as pernas reais."""
    validas = {b: c for b, c in contagens.items() if c >= 3}
    if alocacao == "fixa":
        return {b: cfg.N_PER_BUCKET for b in validas}
    tot = sum(validas.values())
    quota = {b: n_total * c / tot for b, c in validas.items()}
    n = {b: int(np.floor(q)) for b, q in quota.items()}
    resto = sorted(validas, key=lambda b: quota[b] - n[b], reverse=True)
    for b in resto[:n_total - sum(n.values())]:          # maiores restos
        n[b] += 1
    return n


def main():
    global _SEM_PONTOS
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=min(8, mp.cpu_count()))
    ap.add_argument("--n", type=int, default=150, help="sinteticas por rota (alocacao proporcional)")
    ap.add_argument("--alocacao", choices=["proporcional", "fixa"], default="proporcional")
    ap.add_argument("--sem_pontos", action="store_true", help="grava so os metadados")
    ap.add_argument("--saida", default=None)
    args = ap.parse_args()
    _SEM_PONTOS = args.sem_pontos
    OUT_DIR = Path(args.saida) if args.saida else (
        cfg.ROOT / "dados_sinteticos")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    shape_models = load_shape_models(cfg.SHAPE_PATH)
    configs = build_configs(shape_models, cfg.LINHAS_CIRCULARES, cfg.LINHAS_IDA_VOLTA)

    tasks = []
    for (linha, sentido, shape_id, _s, _e) in configs:
        r = build_route_legs(linha, sentido, shape_models, configs)
        shape, _route = load_route(linha, shape_id)
        tag = f"{linha}_{sentido}"
        _ROUTES[tag] = dict(db=r["db"], shape=shape, L=float(shape.total_length),
                            is_loop=r["meta"]["is_loop"], sigma=r["meta"]["sigma_lateral_m"],
                            linha=linha, sentido=sentido, faixas=r["meta"]["faixas"])
        st = pd.to_datetime(r["db"]["start_dt"])
        h_real = (st.dt.hour + st.dt.minute / 60 + st.dt.second / 3600).to_numpy()
        _ROUTES[tag]["horas"] = {b: h_real[(r["db"]["hour_bucket"] == b).to_numpy()]
                                 for b in r["meta"]["faixas"]}
        rng = np.random.default_rng(cfg.SYNTH_SEED + len(tasks))
        cont = {b: int((r["db"]["hour_bucket"] == b).sum()) for b in r["meta"]["faixas"]}
        for b, c in cont.items():
            if 0 < c < 3:
                print(f"  {tag}: faixa {b} sem pernas suficientes ({c}), nao gerada")
        for bucket, n_b in alocar(cont, args.n, args.alocacao).items():
            for i in range(n_b):
                tasks.append((tag, bucket, i, int(rng.integers(1_000_000_000))))
        print(f"{tag}: {len(r['db'])} pernas | {sum(c > 0 for c in cont.values())} faixas | "
              f"sigma lateral {r['meta']['sigma_lateral_m']:.1f} m", flush=True)

    print(f"\nGerando {len(tasks)} viagens sinteticas (K={cfg.SYNTH_K}, "
          f"alpha0={cfg.SYNTH_ALPHA0}, span={cfg.SPAN_RATIO})...", flush=True)
    try:
        mp.set_start_method("fork", force=True)
    except (RuntimeError, ValueError):
        pass
    with mp.Pool(processes=max(1, args.procs)) as pool:
        out = [x for x in pool.map(_gerar, tasks, chunksize=2) if x is not None]

    frames = [f for f, _ in out]
    metas = [m for _, m in out]
    meta_df = pd.DataFrame(metas)
    meta_df.to_csv(OUT_DIR / "metadados_sinteticos.csv", index=False, float_format="%.4f")
    if _SEM_PONTOS:
        print(f"{len(meta_df)} viagens (so metadados) -> {OUT_DIR / 'metadados_sinteticos.csv'}")
        return

    all_pts = pd.concat(frames, ignore_index=True)
    for linha in cfg.LINHAS:
        sub = all_pts[all_pts["linha"] == linha]
        path = OUT_DIR / f"synthetic_fulltrip_dirichlet_{linha}.csv.gz"
        sub.to_csv(path, index=False, float_format="%.7f", compression="gzip")
        n_v = sub["synth_leg_id"].nunique()
        print(f"  {linha}: {n_v} viagens, {len(sub):,} pontos -> {path.name}")
    print(f"\nTotal: {meta_df.shape[0]} viagens | {len(all_pts):,} pontos")
    print(f"Metadados: {OUT_DIR / 'metadados_sinteticos.csv'}")


if __name__ == "__main__":
    main()
