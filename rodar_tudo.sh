#!/bin/sh
# Roda o pipeline completo a partir das rotas descobertas (pipeline/discovered_route_*.csv):
# dados, banco de viagens, conjunto sintetico, verificacao, analises e figuras.
# Requer os brutos em dados_brutos/ (ou SINTETIC_RAW_DIR / SINTETIC_SHAPE_PATH).
# Para refazer tambem as rotas: python 02_descobrir_rota.py && python 02b_corrigir_050_volta.py
set -e
cd "$(dirname "$0")/pipeline"
PY=${PY:-python}
PROCS=${PROCS:-8}

echo "== dados";        $PY 01_baixar_dados.py
echo "== viagens";      $PY legs.py
echo "== sinteticos";   $PY 04_gerar_sinteticos.py --procs "$PROCS"
echo "== verificacao";  $PY verificacao_dirichlet.py
echo "== analise";      $PY 05_analise.py
echo "== desempenho";   $PY 06_desempenho.py
echo "== apresentacao"; $PY ../apresentacao/figuras_apresentacao.py
echo "FIM"
