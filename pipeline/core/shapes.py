"""Carregamento e indexação dos shapes oficiais."""

import json
import lzma
from pathlib import Path

import numpy as np
import pandas as pd

from config import M_PER_DEG_LAT
from core.geometry import meters_per_deg_lon


def load_shape_models(xz_path: Path) -> dict:
    """
    Lê o arquivo de shapes (JSON comprimido) e retorna um dicionário
    indexado por (linha, shape_id) com a geometria em coordenadas
    métricas locais (pronta para projeção).

    Cada modelo contém:
      xs, ys        : vértices em metros
      ax..by        : extremos de cada segmento
      seg_len       : comprimento de cada segmento
      cum_len       : comprimento acumulado (para posição ao longo do shape)
      m_per_deg_lon : fator de conversão local
    """
    with lzma.open(xz_path, "rb") as f:
        obj = json.loads(f.read().decode("utf-8", errors="replace").strip())

    df = pd.DataFrame(obj).rename(
        columns={"SHP": "shape_id", "LAT": "lat", "LON": "lon", "COD": "linha"}
    )
    df["lat"] = pd.to_numeric(
        df["lat"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )
    df["lon"] = pd.to_numeric(
        df["lon"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )
    df["linha"] = df["linha"].astype(str).str.strip().str.zfill(3)
    df["shape_id"] = df["shape_id"].astype(str).str.strip()
    df = df.dropna(subset=["lat", "lon"]).copy()

    models = {}
    for (linha, shape_id), g in df.groupby(["linha", "shape_id"]):
        g = g.reset_index(drop=True)
        lons = g["lon"].to_numpy(dtype=float)
        lats = g["lat"].to_numpy(dtype=float)
        if len(lons) < 2:
            continue

        m_lon = meters_per_deg_lon(float(np.nanmean(lats)))
        xs = lons * m_lon
        ys = lats * M_PER_DEG_LAT
        seg = np.sqrt((xs[1:] - xs[:-1]) ** 2 + (ys[1:] - ys[:-1]) ** 2)
        cum = np.concatenate([[0.0], np.cumsum(seg)])

        models[(linha, shape_id)] = {
            "linha": linha,
            "shape_id": shape_id,
            "lons": lons,
            "lats": lats,
            "xs": xs,
            "ys": ys,
            "ax": xs[:-1], "ay": ys[:-1],
            "bx": xs[1:],  "by": ys[1:],
            "seg_len": seg,
            "cum_len": cum,
            "m_per_deg_lon": m_lon,
            "length_km": float(cum[-1]) / 1000.0,
        }
    return models


def line_shapes(shape_models: dict, linha: str) -> list:
    """Retorna os shape_ids de uma linha, ordenados."""
    return sorted(sid for (l, sid) in shape_models.keys() if l == linha)


def shape_terminal_points(model: dict):
    """Ponto inicial e final de um shape em [lon, lat]."""
    return (
        [float(model["lons"][0]), float(model["lats"][0])],
        [float(model["lons"][-1]), float(model["lats"][-1])],
    )
