"""
functional_synthesis_dirichlet.py
=================================
Sintese de pernas com PESOS ALEATORIOS DE DIRICHLET (versao corrigida).

Base: Burzacchi et al. (2024), arXiv:2410.12514 (FDASynthesis), adaptado ao
projeto. Para uma perna sintetica:

  1. Sorteia uma perna REAL como ancora (mesma faixa horaria). A ancora define o
     dominio de alinhamento (tau) e a referencia de fase. Ela NAO entra nas medias.
  2. Vizinhos: TODAS as pernas da faixa com extensao S entre S_ancora/2 e 2*S_ancora
     (correcao 5: sem subamostragem). Cada um e alinhado a ancora por DP e recebe
     a distancia D = ||q_ancora - q_vizinho||_2 (norma, nao o quadrado; correcao 1).
  3. Parametros de concentracao (Eq. 13): alpha_k = alpha0 * exp(-D_k) / sum_n exp(-D_n).
  4. Pesos: p ~ Dirichlet(alpha) (Eq. 14), sorteados a cada sintese.
  5. SRVF sintetica: q_syn = sum_k p_k q_k (Eq. 15, media ponderada das SRVFs alinhadas).
  6. Forma: gamma_syn = inversa da SRVF (renormalizada para 0 -> 1).
  7. Duracao, extensao e ponto inicial vem SOMENTE dos vizinhos (correcoes 2 e 3):
        T_syn = sum_k p_k T_k,  S_syn = sum_k p_k S_k,  s0_syn = sum_k p_k s0_k
     (para rotas circulares, s0 e media circular modulo L).
  8. Trajetoria a 1 Hz: s(t) = s0_syn + S_syn * gamma_syn(t / T_syn); velocidade
     limitada a MAX_SPEED_MS com reintegracao.

As distancias de cada ancora (NeighbourCache) sao calculadas uma vez e reutilizadas
por quantas sinteses e combinacoes (K, alpha0) forem preciso.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config as cfg
from functional_synthesis import dp_lote_em_blocos, inverse_srvf
from legs import HOUR_BUCKETS
from synthetic_bus_trajectory import RouteShape, project_planar, unproject_planar

KMAX_CACHE = 16          # vizinhos guardados por ancora (K maximo testado)


@dataclass
class NeighbourCache:
    anchor_id: str
    hour_bucket: str
    tau: np.ndarray
    n_candidates: int
    ids: list            # K_MAX vizinhos mais proximos, ordenados por D
    D: np.ndarray        # norma L2 entre SRVFs alinhadas
    regs: np.ndarray     # (K, n_grid) SRVFs dos vizinhos alinhadas a ancora
    T: np.ndarray
    S: np.ndarray
    s0: np.ndarray


@dataclass
class SynthResult:
    anchor_id: str
    hour_bucket: str
    t: np.ndarray        # s, grade de 1 Hz
    s: np.ndarray        # posicao ao longo da rota (m), sem limite de L
    v: np.ndarray        # velocidade (m/s)
    T: float
    S: float
    s0: float
    weights: np.ndarray  # p_k
    alpha: np.ndarray
    neighbour_ids: list
    D: np.ndarray        # distancias dos vizinhos usados
    n_candidates: int
    clip_frac: float     # fracao do tempo em que o teto de velocidade foi aplicado
    gamma: np.ndarray    # curva gamma sintetica (sobre tau), para metricas de privacidade


def build_neighbour_cache(db: pd.DataFrame, anchor: pd.Series, span_ratio: float,
                          kmax: int = KMAX_CACHE) -> NeighbourCache:
    """Alinha a ancora a todas as pernas candidatas e guarda os kmax mais proximos."""
    tau = anchor["tau_grid"]
    q1 = anchor["q"]
    S_a = anchor["S"]
    cand = db[(db["hour_bucket"] == anchor["hour_bucket"])
              & (db["leg_id"] != anchor["leg_id"])
              & (db["S"] >= S_a / span_ratio) & (db["S"] <= S_a * span_ratio)]
    if len(cand) == 0:
        return None

    # DP em lote: identico a dp_align candidata a candidata (verificacao C0)
    costs, regs = dp_lote_em_blocos(q1, np.vstack(cand["q"].to_numpy()), tau)
    D = np.sqrt(np.maximum(costs, 0.0))            # norma L2, nao o quadrado
    order = np.argsort(D, kind="stable")[:kmax]
    c = cand.iloc[order]
    return NeighbourCache(
        anchor_id=str(anchor["leg_id"]), hour_bucket=str(anchor["hour_bucket"]),
        tau=tau, n_candidates=len(cand),
        ids=c["leg_id"].tolist(), D=D[order], regs=regs[order],
        T=c["T"].to_numpy(float), S=c["S"].to_numpy(float), s0=c["s0"].to_numpy(float),
    )


def _mean_position(p: np.ndarray, s0: np.ndarray, L: float | None) -> float:
    """Media ponderada de posicoes; circular (modulo L) em rotas em laco."""
    if L is None:
        return float(np.dot(p, s0))
    ang = 2 * np.pi * s0 / L
    c, s = np.dot(p, np.cos(ang)), np.dot(p, np.sin(ang))
    return float(np.mod(np.arctan2(s, c) / (2 * np.pi) * L, L))


def synthesize_from_cache(cache: NeighbourCache, k: int, alpha0: float,
                          rng: np.random.Generator, loop_length: float | None = None,
                          max_speed: float = cfg.MAX_SPEED_MS) -> SynthResult:
    """Uma sintese a partir de uma ancora ja alinhada (K e alpha0 livres)."""
    k = min(k, len(cache.ids))
    D_k = cache.D[:k]
    g = np.exp(-D_k)
    alpha = alpha0 * g / g.sum()                     # Eq. 13
    p = rng.dirichlet(alpha)                         # Eq. 14

    q_syn = p @ cache.regs[:k]                       # Eq. 15
    gamma_syn = inverse_srvf(q_syn, cache.tau)
    T = float(np.dot(p, cache.T[:k]))                # correcao 2: so vizinhos
    S = float(np.dot(p, cache.S[:k]))
    s0 = _mean_position(p, cache.s0[:k], loop_length)  # correcao 3

    t_grid = np.arange(0.0, T, 1.0)
    gamma_t = np.interp(t_grid / T, cache.tau, gamma_syn)
    s = s0 + S * gamma_t
    v = np.gradient(s, t_grid) if len(t_grid) > 1 else np.zeros_like(t_grid)
    v = np.maximum(v, 0.0)

    clip_frac = 0.0
    if np.any(v > max_speed):
        clip_frac = float(np.mean(v > max_speed))
        v_clip = np.minimum(v, max_speed)
        # reintegra a posicao com a velocidade limitada (trapezio, dt = 1 s)
        s = s[0] + np.concatenate([[0.0], np.cumsum(0.5 * (v_clip[1:] + v_clip[:-1]))])
        v = v_clip

    return SynthResult(
        anchor_id=cache.anchor_id, hour_bucket=cache.hour_bucket,
        t=t_grid, s=s, v=v, T=T, S=S, s0=s0, weights=p, alpha=alpha,
        neighbour_ids=cache.ids[:k], D=D_k, n_candidates=cache.n_candidates,
        clip_frac=clip_frac, gamma=gamma_syn,
    )


def sample_anchor(db: pd.DataFrame, bucket: str, rng: np.random.Generator) -> pd.Series:
    rows = db[db["hour_bucket"] == bucket]
    return rows.iloc[int(rng.integers(len(rows)))]


def result_to_frame(res: SynthResult, shape: RouteShape, sigma_lateral: float,
                    t0: pd.Timestamp, loop_length: float | None,
                    rng: np.random.Generator) -> pd.DataFrame:
    """Converte a trajetoria 1D em lat/lon com ruido lateral (normal, sigma calibrado)."""
    s_pos = np.mod(res.s, loop_length) if loop_length else np.clip(res.s, 0, shape.total_length)
    lat_s, lon_s = shape.s_to_lonlat(s_pos)
    perp_x, perp_y = shape.heading_at_s(s_pos)
    x_s, y_s = project_planar(lat_s, lon_s, shape.ref_lat, shape.ref_lon)
    lateral = rng.normal(0.0, sigma_lateral, size=len(s_pos))
    x_s = x_s + perp_x * lateral
    y_s = y_s + perp_y * lateral
    lat, lon = unproject_planar(x_s, y_s, shape.ref_lat, shape.ref_lon)
    return pd.DataFrame({
        "t_s": res.t,
        "dt": t0 + pd.to_timedelta(res.t, unit="s"),
        "lat": lat, "lon": lon,
        "s_m": s_pos, "v_ms": res.v,
        "lateral_residual_m": lateral,
    })


def bucket_start_hour(bucket: str, rng: np.random.Generator, faixas: dict | None = None,
                      horas_reais: np.ndarray | None = None) -> float:
    """
    Hora de inicio dentro da faixa (faixas = meta["faixas"] da rota).
    Partida real da faixa sorteada (horas_reais) mais um desvio uniforme de
    +-JITTER_INICIO_MIN, mantido dentro da faixa; sem partidas reais, uniforme
    em [inicio, fim) da faixa.
    """
    a, b = (faixas[bucket][:2] if faixas is not None else HOUR_BUCKETS[bucket])
    if horas_reais is not None and len(horas_reais):
        h = horas_reais[int(rng.integers(len(horas_reais)))]
        h += rng.uniform(-1, 1) * cfg.JITTER_INICIO_MIN / 60
        return float(np.clip(h, a, np.nextafter(b, a)))
    return float(rng.uniform(a, b))

