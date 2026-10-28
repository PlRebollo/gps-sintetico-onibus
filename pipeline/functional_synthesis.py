"""
functional_synthesis.py
=======================
Representacao funcional das viagens e alinhamento elastico (base do gerador).

Inspirado em Burzacchi et al. (2024), arXiv:2410.12514 (FDASynthesis), com as
adaptacoes do projeto:

  - Cada perna (viagem de uma linha, de um terminal ao outro, ou uma volta do
    laco) e representada pela fracao do percurso em funcao da fracao do tempo:
    gamma(tau), tau em [0, 1], gamma(0)=0, gamma(1)=1.
  - As curvas sao comparadas no espaco das SRVFs q = sign(gamma') sqrt|gamma'|,
    onde ||q||_2 = 1 para toda perna. A distancia de AMPLITUDE e a norma L2 de
    q1 - q2 apos o alinhamento otimo (programacao dinamica, DP).
  - Como o artigo, o alinhamento usa so amplitude (epsilon = 1).

Funcoes principais:
  build_leg_curve        pings (t, s) -> gamma(tau), T, S, s0
  srvf / inverse_srvf    transformada e inversa
  dp_align               alinhamento elastico otimo (versao vetorizada)
  build_leg_database     pernas -> tabela funcional (uma linha por perna)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

import config as cfg


# ----------------------------------------------------------------------
# 1. Representacao funcional de cada perna
# ----------------------------------------------------------------------
def _sem_saltos_a_frente(t_obs, s_obs, tol_m=None, k=None):
    """
    Descarta pings isolados projetados a frente: s_i maior em mais de tol_m que
    algum dos k pings seguintes (ex.: rota que passa perto de si mesma). Sem isso
    o maximo acumulado "congela" no ping errado e a curva salta centenas de metros
    em um intervalo. Inversoes pequenas (ruido de GPS) nao sao afetadas.
    """
    tol_m = cfg.SALTO_FRENTE_TOL_M if tol_m is None else tol_m
    k = cfg.SALTO_FRENTE_K if k is None else k
    n = len(s_obs)
    if n < 3:
        return t_obs, s_obs
    prox_min = np.full(n, np.inf)
    for j in range(1, k + 1):
        prox_min[:-j] = np.minimum(prox_min[:-j], s_obs[j:])
    ok = ~(s_obs > prox_min + tol_m)
    ok[0] = True
    return t_obs[ok], s_obs[ok]


def build_leg_curve(t_obs: np.ndarray, s_obs: np.ndarray, n_grid: int = 81):
    """
    Converte os pings (t, s) de uma perna em gamma(tau).

      1. s nao decrescente contra ruido de GPS: descarta pings isolados
         projetados a frente e repete a ultima posicao valida
         (maximo acumulado);
      2. interpolacao PCHIP (monotona) para s(t) continuo;
      3. normalizacao: tau = (t - t0)/T, gamma = (s - s0)/S.

    Retorna dict com tau_grid, gamma, T (s), S (m), s0 (m) ou None.
    """
    order = np.argsort(t_obs)
    t_obs, s_obs = t_obs[order], s_obs[order]
    t_obs, s_obs = _sem_saltos_a_frente(t_obs, s_obs)
    s_smooth = np.maximum.accumulate(s_obs)
    t_unique, idx = np.unique(t_obs, return_index=True)
    s_unique = s_smooth[idx]
    if len(t_unique) < 4:
        return None
    interp = PchipInterpolator(t_unique, s_unique)

    T = t_unique[-1] - t_unique[0]
    t0 = t_unique[0]
    tau_grid = np.linspace(0, 1, n_grid)
    s_grid = interp(t0 + tau_grid * T)
    S = s_grid[-1] - s_grid[0]
    if T <= 0 or S <= 0:
        return None
    gamma = (s_grid - s_grid[0]) / S
    gamma[0], gamma[-1] = 0.0, 1.0
    gamma = np.maximum.accumulate(gamma)
    return {"tau_grid": tau_grid, "gamma": gamma, "T": T, "S": S, "s0": s_grid[0]}


# ----------------------------------------------------------------------
# 2. SRVF e inversa
# ----------------------------------------------------------------------
def srvf(gamma: np.ndarray, tau_grid: np.ndarray) -> np.ndarray:
    """q(tau) = sign(gamma'(tau)) * sqrt(|gamma'(tau)|)."""
    dgamma = np.gradient(gamma, tau_grid)
    return np.sign(dgamma) * np.sqrt(np.abs(dgamma))


def inverse_srvf(q: np.ndarray, tau_grid: np.ndarray) -> np.ndarray:
    """
    gamma' = sign(q) q^2 (>= 0), integrada e renormalizada para gamma(0)=0,
    gamma(1)=1. A renormalizacao e necessaria porque a media de SRVFs nao tem
    norma 1 (a media de vetores unitarios tem norma < 1).
    """
    dgamma = np.maximum(np.sign(q) * q ** 2, 0.0)
    gamma = np.concatenate([[0.0], np.cumsum((dgamma[:-1] + dgamma[1:]) / 2 * np.diff(tau_grid))])
    span = gamma[-1] - gamma[0]
    if span <= 1e-9:
        return np.linspace(0, 1, len(tau_grid))
    gamma = (gamma - gamma[0]) / span
    gamma[0], gamma[-1] = 0.0, 1.0
    return np.maximum.accumulate(gamma)


# ----------------------------------------------------------------------
# 3. Alinhamento elastico otimo (programacao dinamica)
# ----------------------------------------------------------------------
_MOVES = [(di, dj) for di in range(1, 5) for dj in range(1, 5)]  # passos permitidos


def dp_align(q1: np.ndarray, q2: np.ndarray, gamma2: np.ndarray, tau_grid: np.ndarray):
    """
    Encontra a reparametrizacao gamma_w (monotona, 0->0, 1->1) que minimiza
        sum_tau (q1(tau) - (q2 o gamma_w)(tau) sqrt(gamma_w'(tau)))^2 dtau.

    Versao vetorizada: para cada linha i da grade, todas as colunas j sao
    calculadas de uma vez para cada passo (di, dj). O resultado e identico ao
    da versao com laco duplo (ver verificacao_dirichlet.py, teste C0).

    Retorna (custo, q2_registrado, gamma_w). O custo e a soma de quadrados
    (quadrado da norma L2 da diferenca), como no codigo original.
    """
    N = len(tau_grid)
    cost = np.full((N, N), np.inf)
    parent_i = np.full((N, N), -1, dtype=np.int64)
    parent_j = np.full((N, N), -1, dtype=np.int64)
    cost[0, 0] = 0.0
    jj = np.arange(N)

    for i in range(1, N):
        best_tot = np.full(N, np.inf)
        best_pi = np.full(N, -1, dtype=np.int64)
        best_pj = np.full(N, -1, dtype=np.int64)
        for di, dj in _MOVES:
            i0 = i - di
            if i0 < 0 or dj >= N:
                continue
            j = jj[dj:]
            j0 = j - dj
            prev = cost[i0, j0]
            d1 = tau_grid[i] - tau_grid[i0]
            d2 = tau_grid[j] - tau_grid[j0]
            s = np.sqrt(d2 / d1)
            seg = 0.5 * ((q1[i0] - q2[j0] * s) ** 2 + (q1[i] - q2[j] * s) ** 2) * d1
            tot = prev + seg
            sl = slice(dj, None)
            upd = tot < best_tot[sl]          # empate: fica o primeiro movimento
            best_tot[sl] = np.where(upd, tot, best_tot[sl])
            best_pi[sl] = np.where(upd, i0, best_pi[sl])
            best_pj[sl] = np.where(upd, j0, best_pj[sl])
        cost[i] = best_tot
        parent_i[i] = best_pi
        parent_j[i] = best_pj

    path = [(N - 1, N - 1)]
    i, j = N - 1, N - 1
    while (i, j) != (0, 0):
        pi, pj = parent_i[i, j], parent_j[i, j]
        if pi < 0:
            break
        path.append((int(pi), int(pj)))
        i, j = int(pi), int(pj)
    path = path[::-1]

    idx_i = np.array([p[0] for p in path])
    idx_j = np.array([p[1] for p in path])
    gamma_w = np.interp(tau_grid, tau_grid[idx_i], tau_grid[idx_j])
    dgamma_w = np.clip(np.gradient(gamma_w, tau_grid), 1e-6, None)
    q2_registered = np.interp(gamma_w, tau_grid, q2) * np.sqrt(dgamma_w)
    final_cost = np.trapezoid((q1 - q2_registered) ** 2, tau_grid)
    return final_cost, q2_registered, gamma_w


def dp_align_lote(q1: np.ndarray, Q2: np.ndarray, tau: np.ndarray):
    """
    dp_align (acima) para C candidatas de uma vez.
    Q2: (C, N). Retorna (custo_final (C,), q2_registrado (C, N)).
    Mesmas operacoes e mesmo desempate (primeiro movimento) do original.
    """
    C, N = Q2.shape
    cost = np.full((C, N, N), np.inf)
    par_i = np.full((C, N, N), -1, dtype=np.int16)
    par_j = np.full((C, N, N), -1, dtype=np.int16)
    cost[:, 0, 0] = 0.0
    jj = np.arange(N)
    for i in range(1, N):
        best = np.full((C, N), np.inf)
        bpi = np.full((C, N), -1, dtype=np.int16)
        bpj = np.full((C, N), -1, dtype=np.int16)
        for di, dj in _MOVES:
            i0 = i - di
            if i0 < 0 or dj >= N:
                continue
            j = jj[dj:]
            j0 = j - dj
            prev = cost[:, i0, j0]
            d1 = tau[i] - tau[i0]
            d2 = tau[j] - tau[j0]
            s = np.sqrt(d2 / d1)
            seg = 0.5 * ((q1[i0] - Q2[:, j0] * s) ** 2 + (q1[i] - Q2[:, j] * s) ** 2) * d1
            tot = prev + seg
            upd = tot < best[:, dj:]
            best[:, dj:] = np.where(upd, tot, best[:, dj:])
            bpi[:, dj:] = np.where(upd, i0, bpi[:, dj:])
            bpj[:, dj:] = np.where(upd, j0, bpj[:, dj:])
        cost[:, i] = best
        par_i[:, i] = bpi
        par_j[:, i] = bpj

    # caminho de volta, todas as candidatas juntas
    ci = np.full(C, N - 1)
    cj = np.full(C, N - 1)
    pi_hist, pj_hist = [ci.copy()], [cj.copy()]
    ativo = np.ones(C, dtype=bool)
    ar = np.arange(C)
    while ativo.any():
        pi = par_i[ar, ci, cj].astype(np.int64)
        pj = par_j[ar, ci, cj].astype(np.int64)
        ativo = ativo & ~((ci == 0) & (cj == 0)) & (pi >= 0)
        ci = np.where(ativo, pi, ci)
        cj = np.where(ativo, pj, cj)
        pi_hist.append(np.where(ativo, ci, -1))
        pj_hist.append(np.where(ativo, cj, -1))
    PI = np.array(pi_hist).T
    PJ = np.array(pj_hist).T
    costs = np.empty(C)
    regs = np.empty((C, N))
    for c in range(C):
        m = PI[c] >= 0
        ii = PI[c][m][::-1]
        jx = PJ[c][m][::-1]
        gw = np.interp(tau, tau[ii], tau[jx])
        dg = np.clip(np.gradient(gw, tau), 1e-6, None)
        reg = np.interp(gw, tau, Q2[c]) * np.sqrt(dg)
        regs[c] = reg
        costs[c] = np.trapezoid((q1 - reg) ** 2, tau)
    return costs, regs


def dp_lote_em_blocos(q1, Q2, tau, bloco: int = 400):
    """dp_align_lote em blocos de candidatas (controla memoria)."""
    cs, rs = [], []
    for a in range(0, len(Q2), bloco):
        c, r = dp_align_lote(q1, Q2[a:a + bloco], tau)
        cs.append(c)
        rs.append(r)
    if not cs:
        return np.empty(0), np.empty((0, len(tau)))
    return np.concatenate(cs), np.vstack(rs)


# ----------------------------------------------------------------------
# 4. Banco de pernas (tabela funcional)
# ----------------------------------------------------------------------
def build_leg_database(legs_df: pd.DataFrame, n_grid: int = 81,
                       min_pings: int = 10, min_span_m: float = 200.0) -> pd.DataFrame:
    """
    legs_df: colunas leg_id, dt (datetime), s_est (m ao longo da rota),
    hour_bucket. Retorna uma linha por perna valida com gamma, q, T, S, s0.
    """
    rows = []
    for lid, leg in legs_df.groupby("leg_id", sort=False):
        leg = leg.sort_values("dt")
        if len(leg) < min_pings:
            continue
        t_obs = (leg["dt"] - leg["dt"].iloc[0]).dt.total_seconds().to_numpy()
        s_obs = leg["s_est"].to_numpy(dtype=float)
        if s_obs.max() - s_obs.min() < min_span_m:
            continue
        curve = build_leg_curve(t_obs, s_obs, n_grid=n_grid)
        if curve is None:
            continue
        rows.append({
            "leg_id": lid,
            "hour_bucket": leg["hour_bucket"].iloc[0],
            "T": curve["T"], "S": curve["S"], "s0": curve["s0"],
            "gamma": curve["gamma"], "q": srvf(curve["gamma"], curve["tau_grid"]),
            "tau_grid": curve["tau_grid"], "start_dt": leg["dt"].iloc[0],
            "n_pings": len(leg),
        })
    return pd.DataFrame(rows)
