#!/usr/bin/env python3
"""
Roda o experimento completo para todos os datasets em `settings.UCI_DATASET_IDS`
e salva um resumo consolidado em `results/all_uci_summary.json`.

Exemplo:
    python scripts/run_all_datasets.py
    python scripts/run_all_datasets.py --save-support   # grava dc_filter_support.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from configs import settings  # noqa: E402
from dcforest.pipeline import run_uci_experiment  # noqa: E402

SUMMARY_PATH = settings.RESULTS_DIR / "all_uci_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-support",
        action="store_true",
        help="Grava dc_filter_support.json por threshold (arquivo grande; padrão: não grava).",
    )
    parser.add_argument(
        "--skip-violation-check",
        action="store_true",
        help="Pula a verificação de violações das DCs (ETAPA 6.5, só informativa).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_results = []

    for uci_id in settings.UCI_DATASET_IDS:
        try:
            # run_uci_experiment já roda os 3 thresholds (0.01, 0.1, 0.5) por
            # padrão e retorna uma lista com um dict de resultado por threshold.
            results_for_id = run_uci_experiment(
                uci_id,
                save_support=args.save_support,
                check_violations=not args.skip_violation_check,
            )
            all_results.extend(results_for_id)
        except Exception as exc:
            print(f"  [ERRO] Falha ao rodar UCI id={uci_id}: {exc}")
            all_results.append({"uci_id": uci_id, "error": str(exc)})

    # Resumo consolidado: métricas + DCs de todos os datasets em um só lugar.
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResumo de todos os datasets salvo em '{SUMMARY_PATH}'")


if __name__ == "__main__":
    main()
