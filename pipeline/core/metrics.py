"""
Métrica de aderência aos dados GPS.

Como o shape oficial pode ter defeitos, a precisão do modelo é medida
diretamente contra os dados observados:
  coverage_Xm     = % de amostras de GPS a <= X m da rota reconstruída
  mean_gps_dist_m = distância média amostra de GPS -> rota
  p95_gps_dist_m  = percentil 95 (captura fugas localizadas)

Bandas padrão: 5 m, 10 m e 15 m (precisão horizontal de GPS urbano).
O MAE contra o shape oficial é mantido apenas como referência.
"""

import numpy as np
import pandas as pd

from config import (
    M_PER_DEG_LAT, ADHERENCE_THRESHOLD_M, ADHERENCE_THRESHOLDS_M,
)
from core.geometry import distance_matrix_to_polyline, meters_per_deg_lon


def route_as_polyline_model(route: pd.DataFrame) -> dict:
    """Converte a rota reconstruída num 'modelo de polilinha' métrico."""
    lons = route["lon"].to_numpy(dtype=float)
    lats = route["lat"].to_numpy(dtype=float)
    m_lon = meters_per_deg_lon(float(np.nanmean(lats)))
    xs = lons * m_lon
    ys = lats * M_PER_DEG_LAT
    seg = np.sqrt((xs[1:] - xs[:-1]) ** 2 + (ys[1:] - ys[:-1]) ** 2)
    return {
        "ax": xs[:-1], "ay": ys[:-1],
        "bx": xs[1:],  "by": ys[1:],
        "seg_len": seg,
        "m_per_deg_lon": m_lon,
    }


def gps_adherence_metrics(
    route: pd.DataFrame,
    gps: pd.DataFrame,
    thresholds_m=None,
    threshold_m: float = None,
    max_gps_points: int = 20000,
) -> dict:
    """
    Distâncias GPS->rota e métricas de aderência em várias bandas.
    Amostra até max_gps_points para manter o custo controlado.

    Retorna, para cada banda X em thresholds_m, a chave `coverage_Xm`
    (% de amostras de GPS dentro de X metros). `coverage_pct` é a banda principal
    (threshold_m, default 15 m).
    """
    if thresholds_m is None:
        thresholds_m = ADHERENCE_THRESHOLDS_M
    if threshold_m is None:
        threshold_m = ADHERENCE_THRESHOLD_M

    empty = {
        "coverage_pct": np.nan, "mean_gps_dist_m": np.nan,
        "p95_gps_dist_m": np.nan, "n_gps_used": 0,
    }
    for t in thresholds_m:
        empty[f"coverage_{int(round(t))}m"] = np.nan
    if route is None or len(route) < 2 or gps is None or gps.empty:
        return empty

    pts = gps[["lon", "lat"]].dropna()
    if len(pts) == 0:
        return empty
    if len(pts) > max_gps_points:
        pts = pts.sample(n=max_gps_points, random_state=42)

    model = route_as_polyline_model(route)
    d = distance_matrix_to_polyline(pts, model)
    valid = d[np.isfinite(d)]
    if len(valid) == 0:
        return empty

    out = {
        "coverage_pct": 100.0 * float(np.mean(valid <= threshold_m)),
        "mean_gps_dist_m": float(np.mean(valid)),
        "p95_gps_dist_m": float(np.percentile(valid, 95)),
        "n_gps_used": int(len(valid)),
    }
    for t in thresholds_m:
        out[f"coverage_{int(round(t))}m"] = 100.0 * float(np.mean(valid <= t))
    return out


def turns_per_km(route: pd.DataFrame) -> float:
    """
    Soma das variações absolutas de rumo (graus) por km.
    Métrica de ziguezague/linearidade da rota.
    """
    lon = route["lon"].to_numpy(dtype=float)
    lat = route["lat"].to_numpy(dtype=float)
    if len(lon) < 3:
        return np.nan
    m_lon = meters_per_deg_lon(float(np.nanmean(lat)))
    heading = np.degrees(np.arctan2(
        np.diff(lat) * M_PER_DEG_LAT, np.diff(lon) * m_lon
    ))
    dh = np.diff(heading)
    dh = (dh + 180.0) % 360.0 - 180.0
    km = route_length_km_local(route)
    return float(np.abs(dh).sum() / km) if km > 0 else np.nan


def route_length_km_local(route: pd.DataFrame) -> float:
    from core.geometry import route_length_km
    return route_length_km(route)
