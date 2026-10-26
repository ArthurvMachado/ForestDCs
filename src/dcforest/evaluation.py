"""
Cálculo de métricas de classificação. Extraído de `testUCI.py`
(era a ETAPA 8 daquele script) — não depende do threshold do LIMA,
então faz sentido rodar uma única vez por dataset.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .data_loading import section


def compute_metrics(
    y_test: pd.Series,
    y_pred,
    title: str = "ETAPA 8: RESULTADOS (fixos, independem do threshold)",
    verbose: bool = True,
) -> dict[str, float]:
    """Calcula accuracy/precision/recall/F1 e (opcionalmente) imprime um resumo."""
    if verbose:
        section(title)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }

    if verbose:
        print(f"  Acurácia  : {metrics['accuracy']:.4f}")
        print(f"  Precisão  : {metrics['precision']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print(f"  F1-Score  : {metrics['f1_score']:.4f}")

    return metrics