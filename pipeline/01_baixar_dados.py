"""
01_baixar_dados.py
==================
PASSO 1 do pipeline: le os pings GPS brutos de marco/2026 das linhas 020, 030
e 331 e gera um unico parquet com o esquema usado pelo modelo v11:

    veiculo, lat, lon, timestamp, linha, data_arquivo

Fonte: arquivos diarios YYYY_MM_DD_veiculos.json.xz (JSON-lines, comprimido),
que ja estao na maquina em RAW_DIR (copia do portal de dados abertos
https://dadosabertos.c3sl.ufpr.br/curitibaurbs). Nao ha download nesta etapa.

Saida: cache/pings_2026_03.parquet

Observacoes:
  - Os arquivos top10 (*_top10.parquet) NAO sao usados como fonte: sao um
    recorte com poucos veiculos (a linha 331 tem apenas ~7 mil pings la, contra
    ~3,6 mil por dia nos brutos).
  - Filtro de linha feito por substring antes do json.loads (como no
    import_pipeline.py original). Duplicatas (VEIC, DTHR, LAT, LON) sao removidas
    dentro de cada dia.
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BASE, LINHAS, RAW_DIR  # noqa: E402

import io  # noqa: E402
import json  # noqa: E402
import lzma  # noqa: E402

CACHE_DIR = BASE / "cache"
OUT_PATH = CACHE_DIR / "pings_2026_03.parquet"


def read_day(path: Path) -> pd.DataFrame:
    """Le um dia de pings das linhas de interesse."""
    recs = []
    with lzma.open(path) as f:
        for raw in io.TextIOWrapper(f, encoding="utf-8", errors="ignore"):
            line = raw.strip()
            if not line:
                continue
            if not any(f'"COD_LINHA":"{ln}"' in line for ln in LINHAS):
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("COD_LINHA") in LINHAS:
                recs.append(rec)
    if not recs:
        return pd.DataFrame()

    df = pd.DataFrame(recs).drop_duplicates(subset=["COD_LINHA", "VEIC", "DTHR", "LAT", "LON"])
    # coordenadas podem usar virgula decimal (pt-BR)
    lat = df["LAT"].astype(str).str.replace(",", ".", regex=False).astype(float)
    lon = df["LON"].astype(str).str.replace(",", ".", regex=False).astype(float)
    out = pd.DataFrame({
        "veiculo": df["VEIC"].astype(str).str.strip(),
        "lat": lat,
        "lon": lon,
        "timestamp": pd.to_datetime(df["DTHR"], format="%d/%m/%Y %H:%M:%S"),
        "linha": df["COD_LINHA"].astype(str).str.strip().str.zfill(3),
        "data_arquivo": path.name[:10],
    })
    return out


def main():
    files = sorted(RAW_DIR.glob("2026_03_*_veiculos.json.xz"))
    print(f"Arquivos brutos de marco/2026: {len(files)}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with Pool(processes=min(8, len(files))) as pool:
        parts = pool.map(read_day, files)

    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    df = df.sort_values(["linha", "veiculo", "timestamp"]).reset_index(drop=True)
    df.to_parquet(OUT_PATH, index=False)

    print("Pings por linha:")
    print(df.groupby("linha").size().to_string())
    print(f"Total: {len(df):,} pings | veiculos: {df['veiculo'].nunique()} | salvo em {OUT_PATH}")


if __name__ == "__main__":
    main()
