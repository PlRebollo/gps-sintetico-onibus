"""
verificacao_dirichlet.py
========================
Testes automaticos da sintese Dirichlet corrigida e do gerador final.

  C0  dp_align vetorizado = versao de referencia (laco duplo do pipeline anterior)
  C1  amostrador rng.dirichlet tem media e variancia de Dirichlet(alpha)
  C2  pesos da sintese: p >= 0 e sum(p) = 1
  C3  T_syn, S_syn e s0_syn sao exatamente sum(p_k * X_k) sobre os vizinhos
      (a ancora nao entra; correcoes 2 e 3)
  C4  alpha_k = alpha0 * exp(-D_k) / sum exp(-D_n), com D_k a NORMA L2 (correcao 1)
  C5  os vizinhos sao os K mais proximos entre TODOS os candidatos da faixa
      (correcao 5), conferido recalculando o alinhamento de todos eles
  C6  viagens geradas: hora de inicio dentro da faixa horaria do rotulo
      (correcao do rotulo antigo)
  C7  concentracao dos pesos: configuracao anterior x artigo x escolhida x uniforme

Saida: resultados/verificacao_dirichlet.csv (C7) e relatorio impresso.

Uso:
    cd sintetic/pipeline && ../.venv/bin/python verificacao_dirichlet.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import warnings

import numpy as np
import pandas as pd

import config as cfg
from core.shapes import load_shape_models
from core.trips import build_configs
from functional_synthesis import dp_align, srvf
from functional_synthesis_dirichlet import (build_neighbour_cache, sample_anchor,
                                            synthesize_from_cache)
from legs import build_route_legs

warnings.filterwarnings("ignore")
RESULTS = cfg.BASE / "resultados"
OK, FALHA = "OK   ", "FALHA"
TOL = 1e-9


def dp_align_reference(q1, q2, gamma2, tau_grid):
    """Versao de referencia (laco duplo), igual a do pipeline anterior."""
    from functional_synthesis import _MOVES
    N = len(tau_grid)
    cost = np.full((N, N), np.inf)
    pi_ = np.full((N, N), -1)
    pj_ = np.full((N, N), -1)
    cost[0, 0] = 0.0
    for i in range(1, N):
        for j in range(1, N):
            best, bi, bj = np.inf, -1, -1
            for di, dj in _MOVES:
                i0, j0 = i - di, j - dj
                if i0 < 0 or j0 < 0 or not np.isfinite(cost[i0, j0]):
                    continue
                d1 = tau_grid[i] - tau_grid[i0]
                d2 = tau_grid[j] - tau_grid[j0]
                s = np.sqrt(d2 / d1)
                seg = 0.5 * ((q1[i0] - q2[j0] * s) ** 2 + (q1[i] - q2[j] * s) ** 2) * d1
                if cost[i0, j0] + seg < best:
                    best, bi, bj = cost[i0, j0] + seg, i0, j0
            cost[i, j], pi_[i, j], pj_[i, j] = best, bi, bj
    path = [(N - 1, N - 1)]
    i, j = N - 1, N - 1
    while (i, j) != (0, 0) and pi_[i, j] >= 0:
        i, j = int(pi_[i, j]), int(pj_[i, j])
        path.append((i, j))
    path = path[::-1]
    idx_i = np.array([p[0] for p in path])
    idx_j = np.array([p[1] for p in path])
    gamma_w = np.interp(tau_grid, tau_grid[idx_i], tau_grid[idx_j])
    dg = np.clip(np.gradient(gamma_w, tau_grid), 1e-6, None)
    q2r = np.interp(gamma_w, tau_grid, q2) * np.sqrt(dg)
    return np.trapezoid((q1 - q2r) ** 2, tau_grid), q2r


def check(label, ok, detalhe=""):
    print(f"  [{OK if ok else FALHA}] {label}" + (f"  ({detalhe})" if detalhe else ""))
    return ok


def main():
    rng = np.random.default_rng(0)
    resultados = []
    print("=" * 74)
    print("VERIFICACAO DA SINTESE DIRICHLET (versao corrigida)")
    print("=" * 74)

    # C0 ---------------------------------------------------------------
    print("\nC0 alinhamento DP")
    tau = np.linspace(0, 1, 81)
    md = 0.0
    for _ in range(10):
        def curva():
            w = rng.uniform(0.2, 3, 4)
            ph = rng.uniform(0, 6, 4)
            g = tau + sum(0.05 * np.sin(2 * np.pi * (k + 1) * tau + ph[k]) / (k + 1)
                          for k in range(4)) * w[0]
            g = np.maximum.accumulate(np.clip(g, 0, 1))
            return (g - g[0]) / (g[-1] - g[0])
        g1, g2 = curva(), curva()
        q1, q2 = srvf(g1, tau), srvf(g2, tau)
        c_new, qn, _ = dp_align(q1, q2, g2, tau)
        c_ref, qr = dp_align_reference(q1, q2, g2, tau)
        md = max(md, abs(c_new - c_ref), float(np.max(np.abs(qn - qr))))
    resultados.append(check("vetorizado = referencia (custo e curva registrada)",
                            md < TOL, f"max dif = {md:.1e}"))

    # C1 ---------------------------------------------------------------
    print("\nC1 amostrador Dirichlet")
    a = np.array([0.6, 1.2, 2.5, 4.0])
    x = np.random.default_rng(1).dirichlet(a, size=200_000)
    m_th = a / a.sum()
    v_th = m_th * (1 - m_th) / (a.sum() + 1)
    em = np.max(np.abs(x.mean(0) - m_th))
    ev = np.max(np.abs(x.var(0) - v_th) / v_th)
    resultados.append(check("media e variancia de Dirichlet", em < 2e-3 and ev < 2e-2,
                            f"erro media {em:.1e}, erro var rel {ev:.1e}"))

    # dados reais: banco do sentido 030 ida ----------------------------
    shape_models = load_shape_models(cfg.SHAPE_PATH)
    configs = build_configs(shape_models, cfg.LINHAS_CIRCULARES, cfg.LINHAS_IDA_VOLTA)
    db = build_route_legs("030", "ida", shape_models, configs)["db"]
    rng2 = np.random.default_rng(5)
    casos = []
    # as 3 faixas mais populosas da rota (rotulos de faixas.definir_faixas)
    for bucket in db["hour_bucket"].value_counts().index[:3]:
        for _ in range(2):
            anc = sample_anchor(db, bucket, rng2)
            cache = build_neighbour_cache(db, anc, cfg.SPAN_RATIO)
            if cache is not None:
                casos.append((anc, cache))

    # C2 e C3 ----------------------------------------------------------
    print("\nC2/C3 pesos e combinacao dos vizinhos")
    soma_err, neg, err_T, err_S, err_s0 = 0.0, 0, 0.0, 0.0, 0.0
    for anc, cache in casos:
        for _ in range(20):
            r = synthesize_from_cache(cache, cfg.SYNTH_K, cfg.SYNTH_ALPHA0, rng2)
            k = len(r.neighbour_ids)
            soma_err = max(soma_err, abs(r.weights.sum() - 1))
            neg += int(np.any(r.weights < 0))
            idx = db.set_index("leg_id").loc[r.neighbour_ids]
            err_T = max(err_T, abs(r.T - float(np.dot(r.weights, idx["T"].to_numpy(float)))))
            err_S = max(err_S, abs(r.S - float(np.dot(r.weights, idx["S"].to_numpy(float)))))
            err_s0 = max(err_s0, abs(r.s0 - float(np.dot(r.weights, idx["s0"].to_numpy(float)))))
    resultados.append(check("pesos >= 0 e somam 1", soma_err < TOL and neg == 0,
                            f"max |sum-1| = {soma_err:.1e}"))
    resultados.append(check("T, S, s0 = soma ponderada dos vizinhos (ancora fora)",
                            max(err_T, err_S, err_s0) < 1e-6,
                            f"erros: T {err_T:.1e}, S {err_S:.1e}, s0 {err_s0:.1e} m"))

    # C4 ---------------------------------------------------------------
    print("\nC4 kernel de Dirichlet com norma L2")
    err_alpha, err_D = 0.0, 0.0
    for anc, cache in casos:
        for _ in range(5):
            r = synthesize_from_cache(cache, cfg.SYNTH_K, cfg.SYNTH_ALPHA0, rng2)
            g = np.exp(-r.D)
            err_alpha = max(err_alpha, np.max(np.abs(r.alpha - cfg.SYNTH_ALPHA0 * g / g.sum())))
        # D deve ser a raiz do custo DP (norma), nao o custo
        cost_nb, _, _ = dp_align(anc["q"], db.set_index("leg_id").loc[cache.ids[0], "q"],
                                 db.set_index("leg_id").loc[cache.ids[0], "gamma"], anc["tau_grid"])
        err_D = max(err_D, abs(cache.D[0] - np.sqrt(cost_nb)))
    resultados.append(check("alpha = alpha0 e^{-D}/sum e D = sqrt(custo DP)",
                            err_alpha < 1e-9 and err_D < 1e-9,
                            f"erro alpha {err_alpha:.1e}, erro D {err_D:.1e}"))

    # C5 ---------------------------------------------------------------
    print("\nC5 vizinhos = K mais proximos entre todos os candidatos")
    ok_knn = True
    for anc, cache in casos[:3]:
        cand = db[(db["hour_bucket"] == anc["hour_bucket"]) & (db["leg_id"] != anc["leg_id"])
                  & (db["S"] >= anc["S"] / cfg.SPAN_RATIO) & (db["S"] <= anc["S"] * cfg.SPAN_RATIO)]
        Dall = np.array([np.sqrt(dp_align(anc["q"], q2, g2, anc["tau_grid"])[0])
                         for q2, g2 in zip(cand["q"], cand["gamma"])])
        top_true = set(cand["leg_id"].to_numpy()[np.argsort(Dall)[:cfg.SYNTH_K]])
        top_code = set(cache.ids[:cfg.SYNTH_K])
        ok_knn &= (top_true == top_code)
    resultados.append(check("vizinhos identicos a busca exaustiva", ok_knn,
                            f"candidatos por ancora ~ {int(np.mean([c.n_candidates for _, c in casos]))}"))

    # C6 ---------------------------------------------------------------
    print("\nC6 faixa horaria x horario de inicio (arquivos gerados)")
    meta_path = cfg.ROOT / "dados_sinteticos" / "metadados_sinteticos.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path, parse_dates=["inicio"], dtype={"linha": str})
        faixas = {}
        for (linha, sentido, _sid, _s, _e) in configs:
            faixas[(linha, sentido)] = build_route_legs(linha, sentido, shape_models,
                                                        configs)["meta"]["faixas"]
        bad = bad_dia = 0
        for _, m in meta.iterrows():
            a0, b0, fds = faixas[(m["linha"], m["sentido"])][m["hour_bucket"]]
            h = m["inicio"].hour + m["inicio"].minute / 60
            bad += int(not (a0 <= h < b0))
            if fds is not None:
                bad_dia += int((m["inicio"].dayofweek >= 5) != fds)
        resultados.append(check("hora de inicio dentro da faixa do rotulo",
                                bad == 0, f"{bad} de {len(meta)} fora da faixa"))
        resultados.append(check("dia sorteado do tipo da faixa (util / fim de semana)",
                                bad_dia == 0, f"{bad_dia} de {len(meta)} no tipo errado"))
    else:
        print("  (metadados_sinteticos.csv nao encontrado: rode 04_gerar_sinteticos.py)")

    # C7 ---------------------------------------------------------------
    print("\nC7 concentracao dos pesos (media sobre ancoras e sorteios)")
    rows = {}
    configs_c7 = [
        ("anterior K=8 a0=7 (kernel d^2)", 8, 7.0, "d2"),
        ("artigo K=6 a0=7 (kernel D)", 6, 7.0, "D"),
        (f"escolhida K={cfg.SYNTH_K} a0={cfg.SYNTH_ALPHA0} (kernel D)", cfg.SYNTH_K, cfg.SYNTH_ALPHA0, "D"),
    ]
    for label, K, a0, kern in configs_c7:
        maxp, neff, ratio, alpha_m, pr = [], [], [], [], []
        for anc, cache in casos:
            D = cache.D[:K] if kern == "D" else cache.D[:K] ** 2
            g = np.exp(-D)
            alpha = a0 * g / g.sum()
            alpha_m.append(alpha.mean())
            ratio.append(alpha.max() / alpha.min())
            P = np.random.default_rng(3).dirichlet(alpha, size=2000)
            maxp.append(P.max(1).mean())
            neff.append((1 / (P ** 2).sum(1)).mean())
        rows[label] = dict(alpha_medio=np.mean(alpha_m), razao_max_min_alpha=np.median(ratio),
                           peso_max_medio=np.mean(maxp), N_eff=np.mean(neff))
    for label, K in [(f"uniforme K={cfg.SYNTH_K}", cfg.SYNTH_K)]:
        P = np.random.default_rng(4).dirichlet(np.ones(K), size=4000)
        rows[label] = dict(alpha_medio=1.0, razao_max_min_alpha=1.0,
                           peso_max_medio=P.max(1).mean(), N_eff=(1 / (P ** 2).sum(1)).mean())
    tab = pd.DataFrame(rows).T
    print(tab.round(3).to_string())
    tab.to_csv(RESULTS / "verificacao_dirichlet.csv", float_format="%.4f")

    n_ok = sum(resultados)
    print(f"\nResultado: {n_ok} de {len(resultados)} testes passaram.")


if __name__ == "__main__":
    main()
