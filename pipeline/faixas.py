"""
faixas.py
=========
Faixas horarias do gerador.

Separadas por tipo de dia (util / sabado e domingo) e, dentro de cada tipo,
cortadas onde o ritmo da linha muda: particao otima de log(duracao) ao longo do
dia por programacao dinamica, com penalidade BIC e minimo de pernas por faixa
(explicacao em docs/faixas_horarias.md).

FAIXAS_FIXAS (5 faixas gerais) ficam apenas para o estudo de K e alpha0
(03_sweep_parametros.py) e como reserva quando a rota nao tem faixas.

Cada faixa e descrita por (inicio_h, fim_h, fim_de_semana). fim_de_semana = None
significa "qualquer dia".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg

FAIXAS_FIXAS = {
    "madrugada": (0, 6),
    "pico_manha": (6, 9),
    "entre_picos": (9, 16),
    "pico_tarde": (16, 20),
    "noite": (20, 24),
}


def particao_ritmo(h: np.ndarray, y: np.ndarray, min_n: int = cfg.RITMO_MIN_PERNAS,
                   slot_min: int = cfg.RITMO_SLOT_MIN,
                   fator_pen: float = cfg.RITMO_FATOR_PEN) -> list:
    """
    Divide [0, 24) em faixas contiguas (fronteiras multiplas de slot_min) que
    minimizam SSE(y dentro das faixas) + pen * n_faixas, com
    pen = fator * sigma2 * ln(n) (BIC; sigma2 = variancia media dentro de cada hora)
    e cada faixa com >= min_n pernas (faixas vazias nas pontas sao toleradas).
    Retorna as fronteiras em horas, de 0 a 24.
    """
    if len(y) < 2 * min_n:
        return [0.0, 24.0]
    ns = int(24 * 60 / slot_min)
    k = np.minimum((h * 60 / slot_min).astype(int), ns - 1)
    n = np.bincount(k, minlength=ns).astype(float)
    s1 = np.bincount(k, weights=y, minlength=ns)
    s2 = np.bincount(k, weights=y * y, minlength=ns)
    cn, c1, c2 = (np.concatenate([[0], np.cumsum(a)]) for a in (n, s1, s2))
    sig2 = pd.Series(y).groupby(h.astype(int)).var().mean()
    if not np.isfinite(sig2):
        sig2 = np.var(y)
    pen = fator_pen * sig2 * np.log(len(y))
    best = np.full(ns + 1, np.inf)
    arg = np.zeros(ns + 1, dtype=int)
    best[0] = 0.0
    for b in range(1, ns + 1):
        for a in range(b):
            m = cn[b] - cn[a]
            if 0 < m < min_n:
                continue
            sse = (c2[b] - c2[a]) - (c1[b] - c1[a]) ** 2 / m if m > 0 else 0.0
            v = best[a] + sse + pen
            if v < best[b]:
                best[b], arg[b] = v, a
    cortes, b = [], ns
    while b > 0:
        cortes.append(b)
        b = arg[b]
    return [0.0] + [c * slot_min / 60 for c in sorted(cortes)]


def _hora(dt: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(dt)
    return (dt.dt.hour + dt.dt.minute / 60 + dt.dt.second / 3600).to_numpy()


def _rotulo(tipo: str, a: float, b: float) -> str:
    fmt = lambda x: f"{int(x):02d}h{int(round((x % 1) * 60)):02d}"
    return f"{tipo}_{fmt(a)}-{fmt(b)}"


def definir_faixas(db: pd.DataFrame) -> dict:
    """{rotulo: (inicio_h, fim_h, fim_de_semana)} para a rota."""
    h = _hora(db["start_dt"])
    fds = pd.to_datetime(db["start_dt"]).dt.dayofweek.to_numpy() >= 5
    y = np.log(db["T"].to_numpy(float))
    out = {}
    for tipo, m in (("util", ~fds), ("fds", fds)):
        fr = particao_ritmo(h[m], y[m])
        for a, b in zip(fr[:-1], fr[1:]):
            out[_rotulo(tipo, a, b)] = (float(a), float(b), tipo == "fds")
    return out


def rotular(dt: pd.Series, faixas: dict) -> np.ndarray:
    """Rotulo da faixa de cada horario de inicio."""
    h = _hora(dt)
    fds = pd.to_datetime(dt).dt.dayofweek.to_numpy() >= 5
    out = np.empty(len(h), dtype=object)
    for k, (a, b, f) in faixas.items():
        m = (h >= a) & (h < b)
        if f is not None:
            m &= fds == f
        out[m] = k
    return out
