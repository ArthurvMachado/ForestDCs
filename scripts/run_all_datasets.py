#!/usr/bin/env python3
"""
Roda o experimento completo para todos os datasets em `settings.UCI_DATASET_IDS`
e salva um resumo consolidado em `results/all_uci_summary.json`.

Exemplo:
    python scripts/run_all_datasets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from configs import settings  # noqa: E402
from dcforest.pipeline import run_uci_experiment  # noqa: E402

SUMMARY_PATH = settings.RESULTS_DIR / "all_uci_summary.json"


def main() -> None:
    all_results = []

    for uci_id in settings.UCI_DATASET_IDS:
        try:
            # run_uci_experiment já roda os 3 thresholds (0.01, 0.1, 0.5) por
            # padrão e retorna uma lista com um dict de resultado por threshold.
            results_for_id = run_uci_experiment(uci_id)
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
