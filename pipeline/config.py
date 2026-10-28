"""
Configuracao do pipeline de Curitiba (linhas 020, 030 e 331).

Parametros de descoberta de rota e de segmentacao vem do experimento v11
(traje_disc/experimento_v11_separacao/config.py), que e a versao mais recente
do modelo de descoberta. Os caminhos de dados podem ser sobrescritos por
variaveis de ambiente.
"""

import os
from pathlib import Path

# ============================================================
# CAMINHOS
# ============================================================
BASE = Path(__file__).resolve().parent          # pipeline/
ROOT = BASE.parent                              # raiz do repositorio

# arquivos brutos diarios (YYYY_MM_DD_veiculos.json.xz) lidos por 01_baixar_dados.py
RAW_DIR = Path(os.environ.get(
    "SINTETIC_RAW_DIR",
    str(ROOT / "dados_brutos")))
# shapeLinha oficial (JSON comprimido), usado para terminais A/B e para MAE
SHAPE_PATH = Path(os.environ.get(
    "SINTETIC_SHAPE_PATH",
    str(ROOT / "dados_brutos" / "2026_03_10_shapeLinha.json.xz")))

# pings de marco/2026 (gerado por 01_baixar_dados.py):
# veiculo, lat, lon, timestamp, linha, data_arquivo
PATH_POSITIONS = BASE / "cache" / "pings_2026_03.parquet"

# ============================================================
# LINHAS
# ============================================================
# 15 linhas, 25 sentidos de rota
LINHAS_CIRCULARES = ["020", "022", "331",       # shape em loop (inicio = fim)
                     "023", "508"]
LINHAS_IDA_VOLTA = ["030", "203",               # dois shapes (um por sentido)
                    "040", "050", "216", "303", "506", "603", "658", "707"]
LINHAS = LINHAS_CIRCULARES + LINHAS_IDA_VOLTA

# ============================================================
# FILTROS DE QUALIDADE (v11)
# ============================================================
USE_SPEED_FILTER = False
MAX_SPEED_KMH = 120
MIN_DT_S = 1
MIN_TRIP_POINTS = 1
MAX_GAP_S = 900                                 # gap > 15 min inicia nova viagem

# ============================================================
# CORTE POR TERMINAL (v11)
# ============================================================
TERMINAL_NEAR_M = 150.0
TERMINAL_FAR_M = 500.0
SEGMENTATION_STRATEGY = "minpoint"              # corta no ponto mais proximo do terminal
SEGMENT_AT_TERMINALS = True
PER_POINT_FILTER = False
STRAIGHT_MIN = 0.15                             # descarta voltas completas ida+volta

# ============================================================
# SELECAO DE VIAGENS (v11)
# ============================================================
START_RADIUS_M = 2000.0
MAX_INPUT_POINTS = 60000
TRIP_SHAPE_MAX_MEDIAN_M = 200.0

# ============================================================
# DESCOBERTA DE ROTA: configuracao FIXA (usada em todas as linhas)
# Escolhida no estudo de grade do v11 (resultados/estudo_grade_v11.csv):
# foi a combinacao com 100% de execucoes validas nas linhas 020, 030 e 331.
# A grade completa abaixo so e rodada para uma linha se a fixa nao for boa.
# ============================================================
FIXED_SAMPLES = 180
FIXED_MIN_METERS = 10.0
FIXED_INCREASE_METERS = 1.25
FIXED_N_TRIPS = 100
GOOD_COVERAGE_15M = 90.0            # limiar de cobertura para aceitar a rota fixa

# ============================================================
# GRADE DE DESCOBERTA DE ROTA (v11, estudo)
# ============================================================
SAMPLES_LIST = [90, 180, 360]
MIN_METERS_LIST = [5, 10, 15]
MAX_METERS_LIST = [250]
INCREASE_METERS_LIST = [1.1, 1.25, 1.5, 1.75, 2.0]
TRIP_COUNTS = [50, 100, 200, 300, 400]

# ============================================================
# MODELO DE CAMINHADA (v11)
# ============================================================
ARRIVAL_RADIUS_M = 25.0
LOOP_CLOSE_RADIUS_M = 25.0
LOOP_MIN_STEPS = 80
CONSECUTIVE_FAIL_STOP = 10
DIST_FLOOR = 0.0
SMOOTH_ITERATIONS = 2

# ============================================================
# PERNAS (viagens usadas como banco funcional)
# ============================================================
IN_ROUTE_M = 50.0                   # ping a <= 50 m da rota descoberta
MIN_LEG_COVER = 0.8                 # perna precisa cobrir >= 80 % da rota
MIN_LEG_PINGS = 10
LAP_PROGRESS_BAND = (0.8, 1.2)      # laco: progresso total da perna entre 0,8 e 1,2 L (uma volta)
LEG_SPEED_BAND_KMH = (4.0, 45.0)    # velocidade media plausivel da perna

# ============================================================
# SINTESE (Dirichlet, versao corrigida)
# Valores finais definidos pelo estudo 03_sweep_parametros.py
# ============================================================
SYNTH_K = 4                         # escolhido em resultados/escolha_parametros.json
SYNTH_ALPHA0 = 2.0
SPAN_RATIO = 1.05                   # candidatos com S entre S/1,05 e 1,05*S (escolhido no estudo)
N_PER_BUCKET = 30                   # sinteticas por faixa horaria e por rota
SYNTH_SEED = 2026
MAX_SPEED_MS = 25.0                 # teto fisico (90 km/h)
MARCH_DAYS = 31

# ============================================================
# MODELO: MONOTONIZACAO, FAIXAS HORARIAS E HORA DE INICIO
#   - lacos desenrolados pela soma dos deslocamentos com sinal (legs.py)
#   - s(t) monotonizado por ultima observacao (maximo acumulado), apos descartar
#     amostras isoladas projetadas a frente (functional_synthesis.py)
#   - faixas horarias por mudanca de ritmo, por rota, separadas em dia util e
#     sabado/domingo (faixas.py; docs/faixas_horarias.md)
#   - hora de inicio da sintetica = partida real da faixa sorteada + desvio
#     uniforme de +-JITTER_INICIO_MIN (nao gera viagens em horarios sem servico)
# ============================================================
# amostra descartada se estiver > SALTO_FRENTE_TOL_M a frente de alguma das
# SALTO_FRENTE_K amostras seguintes (salto isolado; ver functional_synthesis)
SALTO_FRENTE_TOL_M = 100.0
SALTO_FRENTE_K = 3
JITTER_INICIO_MIN = 7.5
RITMO_MIN_PERNAS = 20               # pernas minimas por faixa na particao por ritmo
RITMO_FATOR_PEN = 2.0               # penalidade = fator * sigma2 * ln(n) por faixa (2 = BIC com media + corte);
                                    # escolhidos por validacao cruzada (docs/faixas_horarias.md)
RITMO_SLOT_MIN = 15                 # fronteiras em multiplos de 15 min

# ============================================================
# ADERENCIA (v11)
# ============================================================
ADHERENCE_THRESHOLDS_M = (5.0, 10.0, 15.0)
ADHERENCE_THRESHOLD_M = 15.0

# ============================================================
# CONSTANTES FISICAS
# ============================================================
M_PER_DEG_LAT = 111_132.0
SEGMENT_CHUNK = 8000
