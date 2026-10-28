"""
legs.py
=======
Banco de PERNAS (viagens) de cada sentido de linha, usado pela sintese.

  - Linhas ida/volta: uma perna = uma viagem genuina entre os terminais
    (mesma selecao da descoberta de rota: sentido correto, inicio a <= 2 km do
    terminal, retilinearidade > 0,15, validacao contra o shape).
  - Linhas circulares (laco): uma perna = uma VOLTA completa. O veiculo e cortado
    a cada passagem pelo terminal (mesma regra minpoint). Como a rota fecha em si
    mesma, a posicao ao longo da rota e tratada modulo o comprimento L.

Cada ping e projetado na rota descoberta (RouteShape.match) e so entra se estiver
a <= IN_ROUTE_M metros dela. Uma perna so vale se cobrir >= MIN_LEG_COVER de L.

Saida: cache/legs/{LINHA}_{SENTIDO}.pkl com dict(legs, db, meta).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import config as cfg
from core.shapes import load_shape_models
from core.trips import (build_configs, build_trip_summary, load_gps_data,
                        select_trips, validate_trips)
from core.segmentation import split_at_terminals, terminal_distance
from faixas import FAIXAS_FIXAS, definir_faixas, rotular
from functional_synthesis import build_leg_database
from route_discovery import discovered_route_csv_path
from synthetic_bus_trajectory import RouteShape

CACHE_DIR = cfg.BASE / "cache" / "legs"

# faixas fixas do pipeline original (as faixas usadas ficam em meta["faixas"])
HOUR_BUCKETS = FAIXAS_FIXAS


def hour_bucket(dt: pd.Series) -> pd.Series:
    hours = pd.to_datetime(dt).dt.hour
    out = pd.Series(index=dt.index, dtype=object)
    for label, (a, b) in HOUR_BUCKETS.items():
        out[(hours >= a) & (hours < b)] = label
    return out


def tag_of(linha: str, sentido: str) -> str:
    return f"{linha}_{sentido}"


def load_route(linha: str, shape_id: str):
    """Rota ativa (discovered_route_*.csv) como RouteShape + comprimento."""
    route = pd.read_csv(discovered_route_csv_path(linha, shape_id, routes_dir=cfg.BASE))
    shape = RouteShape.from_points(route["LAT"].to_numpy(), route["LON"].to_numpy())
    return shape, route


def build_route_legs(linha: str, sentido: str, shape_models: dict, configs: list,
                     force: bool = False) -> dict:
    """Constroi (ou le do cache) o banco de pernas de um sentido."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{tag_of(linha, sentido)}.pkl"
    if path.exists() and not force:
        with open(path, "rb") as f:
            return pickle.load(f)

    circular = linha in cfg.LINHAS_CIRCULARES
    line_cfgs = [c for c in configs if c[0] == linha]
    _, _, shape_id, start, end = next(c for c in line_cfgs if c[1] == sentido)

    # terminais: inicio (e fim, nas ida/volta) de todos os sentidos da linha
    pts = []
    for (_l, _s, _sid, st, en) in line_cfgs:
        pts.append(tuple(st))
        if en is not None:
            pts.append(tuple(en))

    df_pl = load_gps_data(cfg.PATH_POSITIONS, linhas=[linha], split_terminals=None,
                          max_gap_s=cfg.MAX_GAP_S, max_speed_kmh=None,
                          min_dt_s=cfg.MIN_DT_S)
    df_pl = terminal_distance(df_pl, {linha: pts})
    df_pl = split_at_terminals(df_pl, strategy=cfg.SEGMENTATION_STRATEGY,
                               near=cfg.TERMINAL_NEAR_M, far=cfg.TERMINAL_FAR_M)
    ts = build_trip_summary(df_pl, min_trip_points=cfg.MIN_TRIP_POINTS)

    if circular:
        sel = ts[ts["linha"] == linha].copy()          # todas as voltas
    else:
        sel = select_trips(ts, linha, start, end, max_vehicles=None,
                           start_radius_m=cfg.START_RADIUS_M)
        sel = validate_trips(df_pl, sel, shape_models, linha, shape_id,
                             max_median_m=cfg.TRIP_SHAPE_MAX_MEDIAN_M)
    n_trips_sel = len(sel)

    shape, route_df = load_route(linha, shape_id)
    L = float(shape.total_length)

    keys = pl.DataFrame({
        "veiculo": sel["veiculo"].astype(str).tolist(),
        "trip_id": sel["trip_id"].astype("int64").tolist(),
    })
    pings = (df_pl.join(keys, on=["veiculo", "trip_id"], how="inner")
             .select(["veiculo", "trip_id", "timestamp", "lat", "lon"]).to_pandas())
    s_raw, resid = shape.match(pings["lat"].to_numpy(), pings["lon"].to_numpy())
    keep = resid <= cfg.IN_ROUTE_M
    pings = pings[keep].reset_index(drop=True)
    s_raw = s_raw[keep]
    resid = resid[keep]

    legs = pd.DataFrame({
        "leg_id": pings["veiculo"] + "|" + pings["trip_id"].astype(str),
        "dt": pd.to_datetime(pings["timestamp"]),
        "lat": pings["lat"].to_numpy(),
        "lon": pings["lon"].to_numpy(),
        "s_raw": s_raw,
    })
    legs = legs.sort_values(["leg_id", "dt"]).reset_index(drop=True)
    if circular:
        # progresso total de cada perna: soma dos deslocamentos com sinal (-L/2, L/2]
        d = legs.groupby("leg_id")["s_raw"].diff().fillna(0.0).to_numpy()
        d = np.mod(d + L / 2, L) - L / 2
        progresso = pd.Series(d).groupby(legs["leg_id"].to_numpy()).sum()
        lo, hi = cfg.LAP_PROGRESS_BAND
        uma_volta = progresso[(progresso >= lo * L) & (progresso <= hi * L)].index
        n_voltas_ruins = int(len(progresso) - len(uma_volta))
        legs = legs[legs["leg_id"].isin(uma_volta)].reset_index(drop=True)
        # s ao longo de cada volta, desenrolado a partir do primeiro ping da volta
        first = legs.groupby("leg_id")["s_raw"].transform("first").to_numpy()
        # soma dos deslocamentos com sinal: uma amostra 10 m atras da primeira nao
        # vira um salto de uma volta inteira (como em first + mod(s - first, L))
        d = legs.groupby("leg_id")["s_raw"].diff().fillna(0.0).to_numpy()
        d = np.mod(d + L / 2, L) - L / 2
        legs["s_est"] = first + pd.Series(d).groupby(legs["leg_id"].to_numpy()).cumsum().to_numpy()
    else:
        n_voltas_ruins = 0
        legs["s_est"] = legs["s_raw"]
    legs["hour_bucket"] = hour_bucket(legs["dt"]).to_numpy()

    db = build_leg_database(legs[["leg_id", "dt", "s_est", "hour_bucket"]],
                            n_grid=81, min_pings=cfg.MIN_LEG_PINGS, min_span_m=200.0)
    n_before = len(db)
    db = db[db["S"] >= cfg.MIN_LEG_COVER * L].reset_index(drop=True)
    v_leg = 3.6 * db["S"] / db["T"]
    lo_v, hi_v = cfg.LEG_SPEED_BAND_KMH
    db = db[(v_leg >= lo_v) & (v_leg <= hi_v)].reset_index(drop=True)

    # faixas horarias (fixas ou por ritmo + tipo de dia, ver faixas.py)
    faixas = definir_faixas(db)
    db["hour_bucket"] = rotular(db["start_dt"], faixas)
    legs["hour_bucket"] = legs["leg_id"].map(dict(zip(db["leg_id"], db["hour_bucket"])))

    # ruido lateral: mediana do |residuo| / 0.6745 (sigma de uma normal, pings na rota)
    sigma_lat = float(np.median(resid)) / 0.6745 if len(resid) else 5.0

    out = dict(
        legs=legs,
        db=db,
        meta=dict(
            linha=linha, sentido=sentido, shape_id=str(shape_id),
            is_loop=bool(circular), route_length_m=L,
            start=list(map(float, start)),
            n_trips_selecionadas=int(n_trips_sel),
            n_pernas_validas_antes_cobertura=int(n_before),
            n_voltas_descartadas_progresso=int(n_voltas_ruins),
            n_pernas=int(len(db)),
            faixas=faixas,
            n_pings_na_rota=int(len(legs)),
            sigma_lateral_m=sigma_lat,
            frac_pings_na_rota=float(keep.mean()) if len(keep) else np.nan,
        ),
    )
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def all_routes(shape_models: dict):
    """Lista (linha, sentido) de todos os sentidos configurados."""
    configs = build_configs(shape_models, cfg.LINHAS_CIRCULARES, cfg.LINHAS_IDA_VOLTA)
    return configs


if __name__ == "__main__":
    shape_models = load_shape_models(cfg.SHAPE_PATH)
    configs = all_routes(shape_models)
    so = set(sys.argv[1:])                      # opcional: python legs.py 050 331
    for (linha, sentido, _sid, _s, _e) in configs:
        if so and linha not in so:
            continue
        r = build_route_legs(linha, sentido, shape_models, configs, force=True)
        m = r["meta"]
        print(f"{linha}/{sentido}: viagens={m['n_trips_selecionadas']} "
              f"pernas validas={m['n_pernas']} | L={m['route_length_m']/1000:.2f} km | "
              f"sigma_lateral={m['sigma_lateral_m']:.1f} m | pings na rota={m['frac_pings_na_rota']*100:.1f}%")
