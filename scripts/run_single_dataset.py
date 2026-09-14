#!/usr/bin/env python3
"""
Roda o experimento completo (treino + LIMA) para um único dataset da UCI.

Exemplos:
    python scripts/run_single_dataset.py --uci-id 53
    python scripts/run_single_dataset.py --uci-id 53 --thresholds 0.01 0.1 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from configs import settings  # noqa: E402
from dcforest.pipeline import run_uci_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uci-id",
        type=int,
        default=53,
        help="ID do dataset na UCI Machine Learning Repository (padrão: 53, Iris).",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(settings.THRESHOLDS),
        help=f"Thresholds do LIMA a testar (padrão: {settings.THRESHOLDS}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_uci_experiment(args.uci_id, thresholds=tuple(args.thresholds))


if __name__ == "__main__":
    main()
