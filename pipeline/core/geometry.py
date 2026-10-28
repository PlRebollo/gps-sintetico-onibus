"""Utilidades geométricas e utilitários gerais."""

import hashlib
import json

import numpy as np
import pandas as pd

from config import M_PER_DEG_LAT, SEGMENT_CHUNK


def meters_per_deg_lon(lat: float) -> float:
    """Fator de conversão longitude->metros na latitude dada."""
    return M_PER_DEG_LAT * np.cos(np.radians(lat))


def route_length_km(route: pd.DataFrame, lon_col="lon", lat_col="lat") -> float:
    """Comprimento da polilinha (Haversine, km)."""
    if len(route) < 2:
        return 0.0
    lat = route[lat_col].to_numpy(dtype=float)
    lon = route[lon_col].to_numpy(dtype=float)
    R = 6371.0088
    d = np.pi / 180.0
    dphi = (lat[1:] - lat[:-1]) * d
    dlmb = (lon[1:] - lon[:-1]) * d
    a = (
        np.sin(dphi / 2) ** 2
        + np.cos(lat[:-1] * d) * np.cos(lat[1:] * d) * np.sin(dlmb / 2) ** 2
    )
    return float(R * 2 * np.arcsin(np.sqrt(a)).sum())


def run_id_from_params(params: dict) -> str:
    """Identificador determinístico de uma execução a partir dos parâmetros."""
    blob = json.dumps(params, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def distance_matrix_to_polyline(points: pd.DataFrame, model: dict) -> np.ndarray:
    """
    Distância de cada ponto à polilinha mais próxima (em metros).

    model: dicionário com ax, ay, bx, by, seg_len, cum_len, m_per_deg_lon.

    Para cada ponto, projeta ortogonalmente em cada segmento (limitada ao
    segmento) e retorna a menor distância entre todos os segmentos.
    Processamento em blocos para eficiência de memória.
    """
    px = points["lon"].to_numpy(dtype=float) * model["m_per_deg_lon"]
    py = points["lat"].to_numpy(dtype=float) * M_PER_DEG_LAT

    ax, ay = model["ax"], model["ay"]
    bx, by = model["bx"], model["by"]

    n = len(px)
    m = len(ax)
    if n == 0 or m == 0:
        return np.full(n, np.nan)

    best = np.full(n, np.inf)
    px_col = px[:, None]
    py_col = py[:, None]

    # bloco dinâmico: limita a matriz a ~4M elementos (32 MB por array),
    # evitando picos de memória quando há muitos pontos/segmentos
    chunk = max(1, min(m, int(4_000_000 // max(n, 1))))
    for start in range(0, m, chunk):
        end = min(start + chunk, m)
        abx = (bx[start:end] - ax[start:end])[None, :]
        aby = (by[start:end] - ay[start:end])[None, :]
        ab2 = abx * abx + aby * aby
        denom = np.where(ab2 == 0.0, 1.0, ab2)

        t = np.clip(
            ((px_col - ax[start:end][None, :]) * abx
             + (py_col - ay[start:end][None, :]) * aby) / denom,
            0.0, 1.0,
        )
        cx = ax[start:end][None, :] + t * abx
        cy = ay[start:end][None, :] + t * aby
        d = np.sqrt((px_col - cx) ** 2 + (py_col - cy) ** 2)
        best = np.minimum(best, d.min(axis=1))

    best[~np.isfinite(best)] = np.nan
    return best


def route_mae_to_shape_m(route: pd.DataFrame, shape_model: dict) -> float:
    """MAE (m) entre os pontos da rota e o shape oficial (referência)."""
    if route is None or len(route) < 2 or shape_model is None:
        return np.nan
    d = distance_matrix_to_polyline(route[["lon", "lat"]], shape_model)
    valid = d[np.isfinite(d)]
    return float(np.mean(valid)) if len(valid) else np.nan
