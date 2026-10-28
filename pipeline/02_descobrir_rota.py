"""
02_descobrir_rota.py
====================
PASSO 2 do pipeline: descobre a rota percorrida por cada linha/sentido a partir
dos pings reais, com o MODELO v11 (caminhada gulosa guiada pelo destino,
core/model.py).

Configuracao FIXA (config.py: FIXED_*) para todas as linhas. A grade completa do
estudo do v11 so e rodada para um sentido quando a configuracao fixa nao da uma
rota boa (conclui, comprimento entre 0,90 e 1,10 do oficial e cobertura GPS a
15 m >= GOOD_COVERAGE_15M).

Fluxo:
  1. Pings de marco/2026 (cache/pings_2026_03.parquet, passo 01).
  2. Segmentacao por gap de 15 min e, nas linhas ida/volta, corte nos terminais
     (estrategia minpoint).
  3. Viagens genuinas: sentido correto, inicio a <= 2 km do terminal, retilinearidade
     > 0,15 e mediana de distancia ao shape oficial <= 200 m. Linhas circulares:
     uma viagem por veiculo (a mais longa).
  4. Para cada sentido: caminhada v11 com a configuracao fixa, sobre as
     FIXED_N_TRIPS viagens de maior numero de pings.
  5. Escolha: (a) completar, (b) comprimento compativel, (c) cobertura.
  6. Fallback: se a rota fixa nao for boa, roda a grade completa daquele sentido.

Saidas (pipeline/):
  discovered_route_{LINHA}[_{SHP}].csv   rota ativa (LON, LAT) usada pelo gerador
                                         (a 331 usa a rota verificada em campo)
  resultados/rota_v11_{LINHA}_{SHP}.csv  rota vencedora do v11 para cada sentido
  resultados/descoberta_fixa.csv         execucoes da configuracao fixa (e fallback)
  resultados/comparacao_rotas.csv        v11 x rota anterior x verificada

Uso:
    cd sintetic/pipeline
    python 02_descobrir_rota.py [--procs N] [--grade-completa]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import multiprocessing as mp
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd

import config as cfg
from core.shapes import load_shape_models
from core.trips import (  # noqa: E402
    load_gps_data, build_trip_summary, select_trips, validate_trips,
    fetch_trip_points, build_configs,
)
from core.segmentation import terminal_distance, split_at_terminals
from core.model import discover_route
from core.metrics import gps_adherence_metrics, route_as_polyline_model, turns_per_km
from core.geometry import (
    route_length_km, route_mae_to_shape_m, run_id_from_params,
    distance_matrix_to_polyline,
)
from route_discovery import discovered_route_csv_path

warnings.filterwarnings("ignore")

RESULTS_DIR = cfg.BASE / "resultados"
OLD_ROUTES_DIR = cfg.BASE / "rotas_anteriores"
VERIFIED = {"331": cfg.BASE / "verified_routes" / "route_331_verified.csv"}

# Faixa de comprimento aceita para a rota descoberta, relativa ao oficial.
LENGTH_RATIO_BAND = (0.90, 1.10)

_STATE = None  # compartilhado com os processos filhos via fork


def selection_key(row):
    """Criterio de escolha da melhor execucao (maior e melhor)."""
    cov = row["coverage_15m"]
    return (int(row["completed"]), int(row["dentro_faixa"]),
            cov if np.isfinite(cov) else -1.0)


def is_good(row):
    """Rota boa o bastante para dispensar a grade completa."""
    return (bool(row["completed"]) and bool(row["dentro_faixa"])
            and np.isfinite(row["coverage_15m"])
            and row["coverage_15m"] >= cfg.GOOD_COVERAGE_15M)


# ============================================================
# EXECUCAO DE UMA COMBINACAO DE HIPERPARAMETROS
# ============================================================
def _discover(st, min_m, max_m, samples, inc):
    return discover_route(
        df=st["gps"], initial_point=st["start_snap"],
        end_point=st["end_snap"], meters=min_m, angular_samples=samples,
        increase_meters=inc, decrease_meters=1.0 / inc, max_meters=max_m,
        loop_close_radius=cfg.LOOP_CLOSE_RADIUS_M,
        loop_min_steps=cfg.LOOP_MIN_STEPS,
        arrival_radius_m=cfg.ARRIVAL_RADIUS_M,
        consecutive_fail_stop=cfg.CONSECUTIVE_FAIL_STOP,
        dist_floor=cfg.DIST_FLOOR,
        smooth_iterations=cfg.SMOOTH_ITERATIONS,
    )


def run_work(wid):
    """Executa as combinacoes de um item (linha, sentido, volume de viagens)."""
    st = _STATE[wid]
    meta = st["meta"]
    rows, best = [], None

    for (min_m, max_m, samples, inc) in st["combos"]:
        params = dict(linha=meta["linha"], sentido=meta["sentido"],
                      n_trips=meta["n_trips"], meters=min_m,
                      max_meters=max_m, samples=samples, inc=inc)
        t0 = time.perf_counter()
        try:
            route, info = _discover(st, min_m, max_m, samples, inc)
            err = ""
        except Exception as exc:  # registra e segue a grade
            route, info, err = pd.DataFrame(), {"stop_reason": "erro",
                                                "n_steps": 0}, str(exc)
        runtime = time.perf_counter() - t0

        if err:
            adh, km, mae, turns = {}, np.nan, np.nan, np.nan
            completed = False
        else:
            adh = gps_adherence_metrics(route, st["gps"])
            km = route_length_km(route)
            mae = route_mae_to_shape_m(route, st["shape_model"])
            turns = turns_per_km(route)
            completed = bool(info["reached_end"] or info["closed_loop"])

        ref_km = st["ref_km"]
        razao = km / ref_km if (err == "" and np.isfinite(ref_km)) else np.nan
        row = dict(
            run_id=run_id_from_params(params), wid=wid,
            linha=meta["linha"], sentido=meta["sentido"],
            shape_id=meta["shape_id"], n_trips=meta["n_trips"],
            n_trips_used=meta["n_trips_used"], n_vehicles=meta["n_vehicles"],
            n_input_points=meta["n_input_points"],
            angular_samples=int(samples), meters=float(min_m),
            max_meters=float(max_m), increase_meters=float(inc),
            n_route_points=len(route) if err == "" else 0,
            route_length_km=km, ref_km=ref_km, razao_comprimento=razao,
            dentro_faixa=int(np.isfinite(razao) and
                             LENGTH_RATIO_BAND[0] <= razao <= LENGTH_RATIO_BAND[1]),
            reached_end=int(info.get("reached_end", 0)),
            closed_loop=int(info.get("closed_loop", 0)),
            completed=int(completed),
            stop_reason=info.get("stop_reason", "erro"),
            coverage_pct=adh.get("coverage_pct", np.nan),
            coverage_5m=adh.get("coverage_5m", np.nan),
            coverage_10m=adh.get("coverage_10m", np.nan),
            coverage_15m=adh.get("coverage_15m", np.nan),
            mean_gps_dist_m=adh.get("mean_gps_dist_m", np.nan),
            p95_gps_dist_m=adh.get("p95_gps_dist_m", np.nan),
            mae_shape_m=mae, turns_per_km=turns,
            runtime_s=runtime, erro=err,
        )
        rows.append(row)
        key = selection_key(row)
        if err == "" and len(route) and (best is None or key > best[0]):
            best = (key, route, row)

    return {"rows": rows, "best_route": best[1] if best else None,
            "best_row": best[2] if best else None}


# ============================================================
# DADOS E ITENS DE TRABALHO
# ============================================================
def load_segmented_pings(configs):
    """Carrega os pings do mes e corta nos terminais as linhas ida/volta."""
    tmap = {}
    for (linha, _sentido, _sid, start, end) in configs:
        if linha not in cfg.LINHAS_IDA_VOLTA:
            continue
        tmap.setdefault(linha, []).append(tuple(start))
        if end is not None:
            tmap[linha].append(tuple(end))

    df_pl = load_gps_data(
        cfg.PATH_POSITIONS, linhas=cfg.LINHAS, split_terminals=None,
        max_gap_s=cfg.MAX_GAP_S, max_speed_kmh=None, min_dt_s=cfg.MIN_DT_S,
    )
    if cfg.SEGMENT_AT_TERMINALS and tmap:
        df_pl = terminal_distance(df_pl, tmap)
        df_pl = split_at_terminals(
            df_pl, strategy=cfg.SEGMENTATION_STRATEGY,
            near=cfg.TERMINAL_NEAR_M, far=cfg.TERMINAL_FAR_M,
        )
    return df_pl


def snap_points(gps, start, end):
    """Ancora inicio/fim no ping real mais proximo do terminal (ate 500 m)."""
    def _snap(p):
        d = np.sqrt((gps["lon"] - p[0]) ** 2 + (gps["lat"] - p[1]) ** 2) * cfg.M_PER_DEG_LAT
        if d.min() > 500:
            return [float(gps["lon"].median()), float(gps["lat"].median())]
        i = int(d.values.argmin())
        return [float(gps["lon"].iloc[i]), float(gps["lat"].iloc[i])]

    ss = _snap(start)
    es = _snap(end) if end is not None else None
    return ss, es


def build_state(shape_models, configs, df_pl, trip_summary, volumes, combos):
    """Um item de trabalho por (linha, sentido, volume de viagens)."""
    state = {}
    for (linha, sentido, shape_id, start, end) in configs:
        sel_all = select_trips(trip_summary, linha, start, end, max_vehicles=None,
                               start_radius_m=cfg.START_RADIUS_M)
        if sel_all.empty:
            print(f"[skip] {linha}/{sentido}: sem viagens")
            continue
        n_before = len(sel_all)
        sel_all = validate_trips(df_pl, sel_all, shape_models, linha, shape_id,
                                 max_median_m=cfg.TRIP_SHAPE_MAX_MEDIAN_M)
        print(f"[viagens] {linha}/{sentido}: {n_before} candidatas, "
              f"{len(sel_all)} apos validacao contra o shape")
        if sel_all.empty:
            continue

        sel_sorted = sel_all.sort_values("n_points", ascending=False).reset_index(drop=True)
        n_avail = len(sel_sorted)
        seen = set()
        for N in volumes:
            eff = min(N, n_avail)
            if eff in seen:
                continue
            seen.add(eff)
            sel_n = sel_sorted.iloc[:eff]
            gps = fetch_trip_points(
                df_pl, sel_n, start=start, end=end,
                per_point_filter=cfg.PER_POINT_FILTER and (end is not None),
                max_points=cfg.MAX_INPUT_POINTS,
            ).dropna(subset=["lon", "lat"]).drop_duplicates(
                subset=["lon", "lat"]).reset_index(drop=True)
            if len(gps) < 50:
                print(f"[skip] {linha}/{sentido} v{N}: poucos pontos ({len(gps)})")
                continue
            ss, es = snap_points(gps, start, end)
            wid = len(state)
            shape_model = shape_models.get((linha, str(shape_id).strip()))
            state[wid] = dict(
                gps=gps, start_snap=ss, end_snap=es,
                shape_model=shape_model,
                ref_km=float(shape_model["length_km"]) if shape_model else np.nan,
                combos=list(combos),
                meta=dict(linha=linha, sentido=sentido, shape_id=str(shape_id),
                          n_trips=N, n_trips_used=eff,
                          n_vehicles=int(sel_n["veiculo"].nunique()),
                          n_input_points=len(gps)),
            )
            print(f"  [item {wid}] {linha}/{sentido} v{N}: {eff}/{n_avail} viagens | "
                  f"{sel_n['veiculo'].nunique()} veiculos | {len(gps):,} pings", flush=True)
    return state


# ============================================================
# COMPARACAO
# ============================================================
def read_route_csv(path):
    df = pd.read_csv(path)
    return df.rename(columns={"LON": "lon", "LAT": "lat"})[["lon", "lat"]]


def mean_dist_between(route_a, route_b):
    """Distancia media (m) dos pontos de A a polilinha B."""
    model = route_as_polyline_model(route_b)
    d = distance_matrix_to_polyline(route_a[["lon", "lat"]], model)
    valid = d[np.isfinite(d)]
    return float(np.mean(valid)) if len(valid) else np.nan


def compare(best_by_key, state_gps):
    table = []
    for (linha, sentido), (route, row, shape_id) in sorted(best_by_key.items()):
        old_path = OLD_ROUTES_DIR / Path(discovered_route_csv_path(linha, shape_id)).name
        old = read_route_csv(old_path) if old_path.exists() else None
        gps = state_gps[(linha, sentido)]
        verified = read_route_csv(VERIFIED[linha]) if linha in VERIFIED else None
        table.append(dict(
            linha=linha, sentido=sentido, shape_id=shape_id,
            n_trips=row["n_trips"], completed=row["completed"],
            params=f"N={int(row['angular_samples'])} r={row['meters']:g} G={row['increase_meters']:g}",
            v11_km=round(row["route_length_km"], 3),
            v11_razao_oficial=round(row["razao_comprimento"], 3),
            v11_cov15=round(row["coverage_15m"], 2),
            v11_mae_oficial_m=round(row["mae_shape_m"], 2),
            anterior_km=round(route_length_km(old), 3) if old is not None else np.nan,
            anterior_cov15=round(gps_adherence_metrics(old, gps)["coverage_15m"], 2) if old is not None else np.nan,
            dist_v11_para_anterior_m=round(mean_dist_between(route, old), 2) if old is not None else np.nan,
            verificada_km=round(route_length_km(verified), 3) if verified is not None else np.nan,
            dist_v11_para_verificada_m=round(mean_dist_between(route, verified), 2) if verified is not None else np.nan,
            dist_verificada_para_v11_m=round(mean_dist_between(verified, route), 2) if verified is not None else np.nan,
        ))
    return pd.DataFrame(table)


def _juntar_anteriores(novo, path, linhas):
    """Com --linhas, mantem as linhas dos resultados anteriores que nao foram rodadas."""
    if not linhas or not Path(path).exists():
        return novo
    old = pd.read_csv(path, dtype={"linha": str})
    old = old[~old["linha"].isin(linhas)]
    return pd.concat([old, novo], ignore_index=True)


# ============================================================
# MAIN
# ============================================================
def main():
    global _STATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=min(8, mp.cpu_count()))
    ap.add_argument("--grade-completa", action="store_true",
                    help="roda a grade completa do v11 em todos os sentidos (estudo)")
    ap.add_argument("--linhas", nargs="*", default=None,
                    help="so estas linhas (resultados anteriores das demais sao mantidos)")
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fixed_combo = [(cfg.FIXED_MIN_METERS, 250.0, cfg.FIXED_SAMPLES, cfg.FIXED_INCREASE_METERS)]
    full_combos = list(product(cfg.MIN_METERS_LIST, cfg.MAX_METERS_LIST,
                               cfg.SAMPLES_LIST, cfg.INCREASE_METERS_LIST))

    print("Carregando shapes oficiais...")
    shape_models = load_shape_models(cfg.SHAPE_PATH)
    configs = build_configs(shape_models, cfg.LINHAS_CIRCULARES, cfg.LINHAS_IDA_VOLTA)
    if args.linhas:
        configs = [c for c in configs if c[0] in args.linhas]
    print(f"Sentidos: {[(c[0], c[1], c[2]) for c in configs]}")

    print("\nCarregando pings de marco/2026 e segmentando...")
    df_pl = load_segmented_pings(configs)
    trip_summary = build_trip_summary(df_pl, min_trip_points=cfg.MIN_TRIP_POINTS)
    print(f"Pings: {df_pl.height:,} | viagens validas: {len(trip_summary):,}")

    print("\nMontando itens (configuracao fixa, N viagens = %d)..." % cfg.FIXED_N_TRIPS)
    combos = full_combos if args.grade_completa else fixed_combo
    _STATE = build_state(shape_models, configs, df_pl, trip_summary,
                         volumes=[cfg.FIXED_N_TRIPS], combos=combos)
    del df_pl, trip_summary
    work_ids = sorted(_STATE.keys())
    print(f"Itens: {len(work_ids)} | execucoes: {sum(len(_STATE[w]['combos']) for w in work_ids)}")

    try:
        mp.set_start_method("fork", force=True)
    except (RuntimeError, ValueError):
        pass

    t0 = time.perf_counter()
    with mp.Pool(processes=max(1, args.procs)) as pool:
        results = pool.map(run_work, work_ids, chunksize=1)
    print(f"Configuracao fixa concluida em {time.perf_counter() - t0:.1f} s")

    rows = [dict(r, fase="fixa") for res in results for r in res["rows"]]
    fallback = [w for w, res in zip(work_ids, results)
                if res["best_row"] is None or not is_good(res["best_row"])]
    if fallback and not args.grade_completa:
        print(f"\nRota fixa nao boa em {len(fallback)} sentido(s): "
              f"{[(_STATE[w]['meta']['linha'], _STATE[w]['meta']['sentido']) for w in fallback]}")
        print("Rodando a grade completa do v11 nesses sentidos...")
        for w in fallback:
            _STATE[w]["combos"] = full_combos
        with mp.Pool(processes=max(1, args.procs)) as pool:
            fb_results = pool.map(run_work, fallback, chunksize=1)
        for w, res in zip(fallback, fb_results):
            results[work_ids.index(w)] = res
            rows += [dict(r, fase="grade_completa") for r in res["rows"]]

    pd.DataFrame(rows).pipe(_juntar_anteriores, RESULTS_DIR / "descoberta_fixa.csv",
                            args.linhas).to_csv(RESULTS_DIR / "descoberta_fixa.csv",
                                                index=False, float_format="%.6f")

    best_by_key, gps_by_key = {}, {}
    for wid, res in zip(work_ids, results):
        row = res["best_row"]
        if row is None:
            continue
        key = (row["linha"], row["sentido"])
        cur = best_by_key.get(key)
        if cur is None or selection_key(row) > selection_key(cur[1]):
            best_by_key[key] = (res["best_route"], row, _STATE[wid]["meta"]["shape_id"])
            gps_by_key[key] = _STATE[wid]["gps"]

    print("\nRotas escolhidas:")
    for (linha, sentido), (route, row, shape_id) in sorted(best_by_key.items()):
        v11_path = RESULTS_DIR / f"rota_v11_{linha}_{shape_id}.csv"
        route[["lon", "lat"]].rename(columns={"lon": "LON", "lat": "LAT"}).to_csv(
            v11_path, index=False, float_format="%.7f")
        active = Path(discovered_route_csv_path(linha, shape_id, routes_dir=cfg.BASE))
        if linha in VERIFIED:
            active.write_bytes(VERIFIED[linha].read_bytes())
            origem = "verificada em campo"
        else:
            route[["lon", "lat"]].rename(columns={"lon": "LON", "lat": "LAT"}).to_csv(
                active, index=False, float_format="%.7f")
            origem = "v11"
        print(f"  {linha}/{sentido} ({shape_id}): {origem} | completa={bool(row['completed'])} "
              f"| razao={row['razao_comprimento']:.3f} | cov15={row['coverage_15m']:.1f}% "
              f"| N={int(row['angular_samples'])} r={row['meters']:g} G={row['increase_meters']:g}")

    comparacao = compare({k: (v[0], v[1], v[2]) for k, v in best_by_key.items()}, gps_by_key)
    comparacao = _juntar_anteriores(comparacao, RESULTS_DIR / "comparacao_rotas.csv", args.linhas)
    comparacao.to_csv(RESULTS_DIR / "comparacao_rotas.csv", index=False)
    print("\n" + comparacao.to_string(index=False))


if __name__ == "__main__":
    main()
