"""
02b_corrigir_050_volta.py
=========================
Correcao pontual da rota descoberta da 050/volta (shape 1720).

O passo 02 escolhe, entre as execucoes que completam e ficam na faixa de
comprimento, a de maior cobertura a 15 m. Na 050/volta a vencedora
(N=360 r=5 G=2: razao 1,097, cobertura 93,345%) tem um zigue-zague espurio de
~1,5 km entre 6,75 e 8,25 km da rota, dentro de uma caixa de ~150 m. Os onibus
nao percorrem esse trecho: a densidade de pings cai a 1/4 e a projecao "pula"
o zigue-zague, o que gera velocidades falsas > 90 km/h em 87% das pernas reais
(nas outras 24 rotas, <= 2%).

Empatada na cobertura (93,335%, diferenca de 0,01 p.p.) esta a execucao
N=180 r=10 G=1.75, com razao 0,995 e sem o zigue-zague. Este script refaz so
essa execucao e a grava como rota ativa. O criterio do passo 02 nao foi mudado
porque trocaria tambem as rotas das linhas originais (020, 030, 331), quebrando
a comparacao com o pipeline original.

Uso:
    python 02b_corrigir_050_volta.py
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import config as cfg
from core.shapes import load_shape_models
from core.trips import build_configs, build_trip_summary
from route_discovery import discovered_route_csv_path

spec = importlib.util.spec_from_file_location("p02", Path(__file__).with_name("02_descobrir_rota.py"))
p02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p02)

LINHA, SENTIDO = "050", "volta"
COMBO = (10.0, 250.0, 180, 1.75)        # (r, max_m, N, G)


def main():
    shape_models = load_shape_models(cfg.SHAPE_PATH)
    configs = [c for c in build_configs(shape_models, cfg.LINHAS_CIRCULARES, cfg.LINHAS_IDA_VOLTA)
               if c[0] == LINHA and c[1] == SENTIDO]
    df_pl = p02.load_segmented_pings(configs)
    trip_summary = build_trip_summary(df_pl, min_trip_points=cfg.MIN_TRIP_POINTS)
    p02._STATE = p02.build_state(shape_models, configs, df_pl, trip_summary,
                                 volumes=[cfg.FIXED_N_TRIPS], combos=[COMBO])
    (wid,) = p02._STATE
    res = p02.run_work(wid)
    row, route = res["best_row"], res["best_route"]
    shape_id = p02._STATE[wid]["meta"]["shape_id"]
    print(f"{LINHA}/{SENTIDO}: razao={row['razao_comprimento']:.3f} cov15={row['coverage_15m']:.3f}% "
          f"km={row['route_length_km']:.3f}")

    out = route[["lon", "lat"]].rename(columns={"lon": "LON", "lat": "LAT"})
    out.to_csv(p02.RESULTS_DIR / f"rota_v11_{LINHA}_{shape_id}.csv", index=False, float_format="%.7f")
    out.to_csv(discovered_route_csv_path(LINHA, shape_id, routes_dir=cfg.BASE), index=False,
               float_format="%.7f")

    path = p02.RESULTS_DIR / "comparacao_rotas.csv"
    comp = pd.read_csv(path, dtype={"linha": str, "shape_id": str})
    m = (comp["linha"] == LINHA) & (comp["sentido"] == SENTIDO)
    comp.loc[m, "params"] = f"N={COMBO[2]} r={COMBO[0]:g} G={COMBO[3]:g} (corrigida, 02b)"
    comp.loc[m, "v11_km"] = round(row["route_length_km"], 3)
    comp.loc[m, "v11_razao_oficial"] = round(row["razao_comprimento"], 3)
    comp.loc[m, "v11_cov15"] = round(row["coverage_15m"], 2)
    comp.loc[m, "v11_mae_oficial_m"] = round(row["mae_shape_m"], 2)
    comp.to_csv(path, index=False)


if __name__ == "__main__":
    main()
