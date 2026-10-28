"""
Separação alternativa de viagens ida/volta e momentos de parada.

Módulo experimental (v11). Não altera o pipeline do v10: reaproveita
`load_gps_data(..., split_terminals=None)` para obter a base já segmentada
por gap temporal e aplica, por cima, estratégias distintas de corte nos
terminais.

Motivação
---------
O corte do v10 dispara na PRIMEIRA amostra que entra no raio `near` do
terminal (150 m). Consequência: a viagem de ida termina a ~150-300 m de B
(o trecho final de aproximação e a parada no terminal são empurrados para
o segmento seguinte). A nuvem selecionada nunca chega ao terminal e o
modelo encerra a rota antes do fim.

Estratégias implementadas (todas geométricas, via `_d_min` = distância ao
terminal mais próximo da linha):

  baseline   primeira entrada no raio `near` (comportamento do v10)
  minpoint   ponto de MENOR distância ao terminal de cada visita (o
             centro da parada); a viagem que chega termina em B e a que
             parte começa em B            <-- RECOMENDADO (sem velocidade)
  visit_end  último ponto da visita dentro do raio `near` (parada inteira
             fica na viagem que chega)
  visit_mid  ponto médio (em amostras) da visita no raio `near`

Variantes OPCIONAIS que usam velocidade como critério extra (não
recomendadas no caso de amostragem esparsa, pois a velocidade estimada é
ruidosa entre amostras distantes):

  stop       ponto de MENOR velocidade da visita (momento de parada)
  dwell_min  `minpoint`, mas só corta se a visita realmente parar
             (velocidade mínima abaixo de `stop_speed_kmh`)

Uma "visita" é uma sequência contígua de amostras dentro do raio `near`,
em cada (linha, veículo, segmento temporal). Só contam visitas
"genuínas": aquelas em que o veículo já esteve a mais de `far` do
terminal no mesmo segmento (evita cortar a partida inicial do dia).
"""

import numpy as np
import polars as pl

from config import M_PER_DEG_LAT, PATH_POSITIONS, MAX_GAP_S, MIN_DT_S

GROUP_SEG = ["linha", "veiculo", "trip_id"]
GROUP_RUN = ["linha", "veiculo", "trip_id", "_run"]

STRATEGIES = ("baseline", "minpoint", "visit_end", "visit_mid",
              "stop", "dwell_min")


# ============================================================
# CARREGAMENTO SEGMENTADO (usado pelo experimento e pelas análises)
# ============================================================
def load_segmented(linhas, split_terminals=None, strategy="minpoint",
                   max_gap_s=None, max_speed_kmh=None, min_dt_s=None,
                   near=150.0, far=500.0) -> pl.DataFrame:
    """
    Carrega o GPS, segmenta por gap temporal e aplica o corte nos terminais
    conforme `strategy`. É o ponto único de entrada usado pelo experimento e
    pelos scripts de análise, garantindo que nuvem e rota usem a MESMA
    separação.
    """
    from core.trips import load_gps_data

    mg = MAX_GAP_S if max_gap_s is None else max_gap_s
    md = MIN_DT_S if min_dt_s is None else min_dt_s
    if strategy == "baseline" or not split_terminals:
        return load_gps_data(
            PATH_POSITIONS, linhas=linhas, split_terminals=split_terminals,
            max_gap_s=mg, max_speed_kmh=max_speed_kmh, min_dt_s=md,
            terminal_near_m=near, terminal_far_m=far)
    base = load_gps_data(
        PATH_POSITIONS, linhas=linhas, split_terminals=None,
        max_gap_s=mg, max_speed_kmh=max_speed_kmh, min_dt_s=md)
    base = terminal_distance(base, split_terminals)
    return split_at_terminals(base, strategy=strategy, near=near, far=far)


# ============================================================
# DISTÂNCIA AO TERMINAL
# ============================================================
def terminal_distance(df: pl.DataFrame, split_terminals: dict) -> pl.DataFrame:
    """
    Adiciona a coluna `_d_min` = distância (m) ao terminal mais próximo da
    linha de cada amostra. Linhas fora de `split_terminals` ficam nulas.
    """
    rows = []
    for k_, pts_ in split_terminals.items():
        for p in pts_:
            rows.append({"linha": str(k_), "t_lon": float(p[0]),
                         "t_lat": float(p[1])})
    term = pl.DataFrame(rows)

    idx = df.with_row_index("_rid")
    j = (idx.filter(pl.col("linha").is_in([str(k_) for k_ in split_terminals]))
         .join(term, on="linha", how="inner"))
    j = j.with_columns(
        (
            ((pl.col("lon") - pl.col("t_lon")) * M_PER_DEG_LAT
             * (pl.col("lat") * np.pi / 180).cos()) ** 2
            + ((pl.col("lat") - pl.col("t_lat")) * M_PER_DEG_LAT) ** 2
        ).sqrt().alias("d_t")
    )
    mind = j.group_by("_rid").agg(pl.col("d_t").min().alias("_d_min"))
    return (idx.join(mind, on="_rid", how="left").drop("_rid")
            .sort(["linha", "veiculo", "timestamp"]))


# ============================================================
# CORTE POR TERMINAIS
# ============================================================
def split_at_terminals(df: pl.DataFrame, strategy: str = "minpoint",
                       near: float = 150.0, far: float = 500.0,
                       stop_speed_kmh: float = 3.0,
                       min_run_len: int = 1) -> pl.DataFrame:
    """
    Retorna `df` com `trip_id` recortado nos terminais conforme `strategy`.

    Requer `_d_min` (via `terminal_distance`). Além disso, o corte por
    terminal exige que a visita seja "genuína" (veículo já esteve a mais de
    `far` metros no mesmo segmento temporal).
    """
    if strategy == "baseline":
        pass
    g = GROUP_SEG
    out = df.sort(g + ["timestamp"]).with_row_index("_row")

    # A velocidade só é necessária nas variantes opcionais stop/dwell_min.
    # O método recomendado (minpoint) e as demais variantes são puramente
    # geométricos (distância ao terminal), robustos a amostragem esparsa.
    use_speed = strategy in ("stop", "dwell_min")
    base = [
        (pl.col("_d_min") > far).cum_max().over(g).alias("_far"),
        (pl.col("_d_min") < near).alias("_near"),
    ]
    if use_speed:
        base.append(pl.col("speed_kmh").fill_null(float("inf")).alias("_spd"))
    out = out.with_columns(base)
    # run contíguo de "_near" dentro de cada segmento
    chg = (pl.col("_near") != pl.col("_near").shift(1).over(g))
    out = out.with_columns(
        chg.fill_null(True).cast(pl.Int64).cum_sum().over(g).alias("_run")
    )

    drop_extra = [c for c in ("_far", "_near", "_spd", "_run", "_row")
                  if c in out.columns]
    near_df = out.filter(pl.col("_near"))
    if near_df.height == 0 or out.height == 0:
        return out.drop(drop_extra)

    aggs = [
        pl.col("_row").min().alias("i_first"),
        pl.col("_row").max().alias("i_last"),
        pl.len().alias("n_run"),
        pl.col("_far").max().alias("far_ok"),
    ]
    if use_speed:
        aggs.append(pl.col("_spd").min().alias("min_spd"))
    runs = near_df.group_by(GROUP_RUN).agg(aggs)
    dmin = (near_df.with_columns(
                pl.col("_d_min").rank("ordinal").over(GROUP_RUN).alias("_r"))
            .filter(pl.col("_r") == 1)
            .select(GROUP_RUN + [pl.col("_row").alias("i_dmin")]))
    runs = runs.join(dmin, on=GROUP_RUN, how="left")
    if use_speed:
        spd = (near_df.with_columns(
                   pl.col("_spd").rank("ordinal").over(GROUP_RUN).alias("_r"))
               .filter(pl.col("_r") == 1)
               .select(GROUP_RUN + [pl.col("_row").alias("i_stop")]))
        runs = runs.join(spd, on=GROUP_RUN, how="left")

    gen = pl.col("far_ok") & (pl.col("n_run") >= min_run_len)
    if strategy == "baseline":
        i_next = pl.col("i_first")
    elif strategy == "minpoint":
        i_next = pl.col("i_dmin") + 1
    elif strategy == "visit_end":
        i_next = pl.col("i_last") + 1
    elif strategy == "visit_mid":
        i_next = pl.col("i_first") + (pl.col("n_run") - 1) // 2 + 1
    elif strategy == "stop":
        i_next = pl.col("i_stop") + 1
        gen = gen & (pl.col("min_spd") <= stop_speed_kmh)
    elif strategy == "dwell_min":
        i_next = pl.col("i_dmin") + 1
        gen = gen & (pl.col("min_spd") <= stop_speed_kmh)
    else:
        raise ValueError(f"estratégia desconhecida: {strategy}")

    bnd = (runs.filter(gen)
           .select(["linha", "veiculo", "trip_id", i_next.alias("i_next")]))
    nextrows = out.select(["linha", "veiculo", "trip_id",
                           pl.col("_row").alias("i_next")])
    marker = (bnd.join(nextrows, on=g + ["i_next"], how="inner")
              .select("i_next").unique()
              .with_columns(pl.lit(1).alias("_inc")))
    out = out.join(marker, left_on="_row", right_on="i_next", how="left")
    out = out.sort(g + ["timestamp"])
    out = out.with_columns(
        (pl.col("trip_id")
         + pl.col("_inc").fill_null(0).cast(pl.Int64)
         .cum_sum().over(["linha", "veiculo"])).alias("trip_id")
    )
    return out.drop([c for c in ("_far", "_near", "_spd", "_run", "_inc", "_row")
                     if c in out.columns])
