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
# Pasta a partir da qual `java -jar ...` é executado. É nela que o LIMA
# escreve `output.txt` ao final da execução.
LIMA_PROJECT_DIR = PROJECT_ROOT
JAR_RELATIVE_PATH = "external/LIMA/target/LIMA-0.1.jar"  # relativo a LIMA_PROJECT_DIR
OUTPUT_FILENAME = "output.txt"

DEFAULT_THRESHOLD = 0.1
THRESHOLDS = (0.01, 0.1, 0.5)
SAMPLE_SIZE = None  # None -> usa len(x_train) automaticamente
SEED = 42
TARGET_COLUMN = None  # None para não incluir o rótulo no CSV do treino
TIMEOUT_S = None  # sem limite por padrão (roda sobre o treino inteiro)
DEBUG = False  # imprime stdout/stderr do Java e mantém o CSV gerado em data/interim

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

# ── Datasets UCI usados no lote (era o `ids` hardcoded em allUCIs.py) ──
UCI_DATASET_IDS = (53, 109, 45, 42, 39, 850, 267, 545, 78)
