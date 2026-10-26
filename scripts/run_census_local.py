#!/usr/bin/env python3
"""
Roda o pipeline no Census Income (adult.data) lido de um arquivo local,
sem passar pelo ucimlrepo. Coloque este arquivo em `scripts/` do repositório.

Exemplo:
    python scripts/run_census_local.py --adult-data /caminho/adult.data \
        --n 10000 --thresholds 0.0001 0.001 --tag census10k

Observações:
  * Nomes de coluna com "_" (e não "-"): os regex de DC usam \\w+.
  * "?" vira a categoria "NA".
  * Alvo = 1ª classe vista (mesma regra do load_dataset do repo).
  * Resultados em results/<tag>/. O dc_filter_support.json é grande; com o
    dataset inteiro e vários thresholds ele pode encher o disco.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from configs import settings  # noqa: E402
from dcforest import pipeline  # noqa: E402
from dcforest.data_loading import section  # noqa: E402

COLS = ["age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income"]

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--adult-data", required=True, help="caminho do adult.data")
ap.add_argument("--n", type=int, default=0, help="amostra de linhas (0 = tudo)")
ap.add_argument("--thresholds", type=float, nargs="+", required=True)
ap.add_argument("--tag", default="census")
a = ap.parse_args()


def load_local_census(uci_id):
    section("ETAPA 1: CARREGANDO O DATASET (Census Income local)")
    df = pd.read_csv(a.adult_data, header=None, names=COLS,
                     skipinitialspace=True, na_values="?")
    df = df.fillna("NA")
    if a.n:
        df = df.sample(n=a.n, random_state=42).reset_index(drop=True)
    first = df["income"].unique()[0]
    df["income"] = (df["income"] == first).astype(int)
    print(f"  Alvo (=1) = {first!r}; {len(df):,} linhas, {len(df.columns)} colunas")
    return df, "income"


pipeline.load_dataset = load_local_census
settings.DEBUG = False
settings.RESULTS_DIR = settings.PROJECT_ROOT / "results" / a.tag
settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
pipeline.run_uci_experiment(a.tag, thresholds=tuple(a.thresholds))
