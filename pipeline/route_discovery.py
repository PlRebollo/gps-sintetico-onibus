"""
route_discovery.py
==================
Utilitarios de rota descoberta. O algoritmo de descoberta em si esta em
core/model.py (modelo v11: caminhada gulosa guiada pelo destino). Este modulo
so cuida de nomes de arquivo e de carregar as rotas como shapes para o
gerador sintetico.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import BASE

# linha -> shapes (SHP) usados. Mesma convencao do shapeLinha oficial.
# 030 tem dois sentidos: 5246 = ida (A->B), 5247 = volta (B->A).
LINE_SHAPE_IDS = {
    "020": ["4091"],
    "022": ["4146"],
    "030": ["5246", "5247"],
    "203": ["3855", "3856"],
    "331": ["4672"],
    # linhas (shapeLinha de 2026-03-10)
    "023": ["4911"],
    "508": ["3900"],
    "040": ["1717", "1718"],
    "050": ["1719", "1720"],
    "216": ["1785", "2778"],
    "303": ["2585", "3617"],
    "506": ["2601", "2602"],
    "603": ["3155", "3156"],
    "658": ["2114", "2115"],
    "707": ["5183", "5184"],
}


def discovered_route_csv_path(line: str, shp_id: str, routes_dir: Path = BASE) -> str:
    """Caminho do CSV da rota descoberta (um por sentido nas linhas ida/volta)."""
    if len(LINE_SHAPE_IDS.get(line, [])) > 1:
        return str(Path(routes_dir) / f"discovered_route_{line}_{shp_id}.csv")
    return str(Path(routes_dir) / f"discovered_route_{line}.csv")


def load_discovered_shapes(line: str, routes_dir: Path = BASE) -> dict:
    """
    Carrega as rotas descobertas de uma linha como {shp_id: RouteShape},
    o mesmo formato usado pelo gerador sintetico.
    """
    from synthetic_bus_trajectory import RouteShape

    shapes = {}
    for shp_id in LINE_SHAPE_IDS[line]:
        route = pd.read_csv(discovered_route_csv_path(line, shp_id, routes_dir))
        shapes[shp_id] = RouteShape.from_points(route["LAT"].to_numpy(),
                                                route["LON"].to_numpy())
    return shapes
