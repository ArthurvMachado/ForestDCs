"""
Configuração central do projeto. Antes, essas constantes estavam
espalhadas como variáveis de módulo em `testUCI.py` e `generateDCs.py`.
Centralizá-las aqui facilita rodar variações do experimento (outro
max_depth, outros thresholds do LIMA etc.) sem tocar no código-fonte.
"""

from __future__ import annotations

from pathlib import Path

# ── Diretórios do projeto ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
RESULTS_DIR = PROJECT_ROOT / "results"

for _dir in (RAW_DATA_DIR, INTERIM_DATA_DIR, RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ── LIMA (descoberta de Denial Constraints) ────────────────────
# Pasta usada só para localizar o JAR (JAR_RELATIVE_PATH é relativo a ela).
# O Java NÃO roda mais aqui: cada execução usa uma pasta temporária, porque o
# LIMA sempre escreve `output.txt` no diretório corrente. Assim a raiz do projeto
# fica limpa e execuções simultâneas não sobrescrevem a saída uma da outra.
LIMA_PROJECT_DIR = PROJECT_ROOT
JAR_RELATIVE_PATH = "external/LIMA/target/LIMA-0.1.jar"  # relativo a LIMA_PROJECT_DIR
OUTPUT_FILENAME = "output.txt"  # nome fixo no código do LIMA

# Heap máximo do Java (ex.: "6g", "12g"). None -> padrão da JVM (1/4 da RAM).
# Aumente se aparecer a mensagem de falta de memória do LIMA.
JAVA_XMX: str | None = None

DEFAULT_THRESHOLD = 0.1
THRESHOLDS = (0.01, 0.1, 0.5)
SAMPLE_SIZE = None  # None -> usa len(x_train) automaticamente
SEED = 42
TARGET_COLUMN = None  # None para não incluir o rótulo no CSV do treino
TIMEOUT_S = None  # sem limite por padrão (roda sobre o treino inteiro)
DEBUG = True  # imprime stdout/stderr do Java e mantém o CSV gerado em data/interim

# Grava `dc_filter_support.json` (suporte por folha/instância) em cada pasta de threshold.
# Fica MUITO grande em datasets grandes, então o padrão é não gravar. Ligue com a flag
# `--save-support` dos scripts, ou trocando aqui.
SAVE_SUPPORT = False

# Verificação de violações das DCs no treino/teste inteiros (ETAPA 6.5). É só
# informativa -- não influencia as predições -- e custa tempo (quadrático em n).
# Desligue com a flag `--skip-violation-check` dos scripts, ou trocando aqui.
CHECK_VIOLATIONS = True

# ── Random Forest ───────────────────────────────────────────────
N_ESTIMATORS = 100
MAX_DEPTH = 8
RANDOM_STATE = 42
TEST_SIZE = 0.2

POSITIVE_LABEL = 1  # Alvo
NEGATIVE_LABEL = 0  # Não Alvo
LABEL_NAMES = {0: "Não Alvo", 1: "Alvo"}

DROP_COLUMNS: list[str] = []
CATEGORICAL_COLUMNS: list[str] = []

# ── Filtro de tuplas por DC (só predicados = e !=) ──────────────
# Tolerância: uma tupla de treino é removida da folha se violar MAIS que
# `max_violations` DCs junto com a instância classificada. 0 = qualquer
# violação remove. Cada valor da grade gera uma rodada de predição.
# (Não confundir com THRESHOLDS, que é o threshold de DC aproximada do LIMA.)
DC_MAX_VIOLATIONS_GRID = (0, 1, 2)

# O que fazer se TODAS as tuplas de uma folha forem removidas:
#   "skip"          -> a árvore não vota para aquela instância
#   "keep_original" -> a árvore mantém a distribuição original da folha
DC_EMPTY_LEAF_POLICY = "skip"

# ── Datasets UCI usados no lote (era o `ids` hardcoded em allUCIs.py) ──
# UCI_DATASET_IDS = (53, 109, 45, 42, 39, 850, 267, 545, 78)
UCI_DATASET_IDS = (2, 17, 52, 73, 94, 151, 222, 267, 350) # Somente Targets Binários