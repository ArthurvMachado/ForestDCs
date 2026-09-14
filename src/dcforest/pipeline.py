"""
Orquestração do experimento completo para um único dataset da UCI:
carrega -> treina -> extrai suporte de folhas -> roda o LIMA por
threshold -> salva métricas e DCs.

Era a função `testUCI()` em `testUCI.py`; a diferença principal é que os
resultados agora vão para `results/uci_<id>/th_<threshold>/` em vez de
arquivos soltos `uci_<id>_th_<threshold>_*.json` na raiz de `results/`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from configs import settings
from .data_loading import load_dataset, prepare_features, section, split_dataset
from .dc_discovery import generate_dcs_for_support
from .evaluation import compute_metrics
from .forest import TraceableRandomForestClassifier


def train_model(x_train_encoded, y_train) -> TraceableRandomForestClassifier:
    """Treina a Random Forest rastreável com os hiperparâmetros de settings."""
    section("ETAPA 4: TREINANDO O MODELO")

    clf = TraceableRandomForestClassifier(
        n_estimators=settings.N_ESTIMATORS,
        max_depth=settings.MAX_DEPTH,
        bootstrap=True,
        random_state=settings.RANDOM_STATE,
        n_jobs=-1,
    )

    start = time.time()
    clf.fit(x_train_encoded, y_train)
    print(f"  Treinado em {time.time() - start:.2f}s")

    return clf


def _dataset_results_dir(uci_id) -> Path:
    path = settings.RESULTS_DIR / f"uci_{uci_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _threshold_results_dir(uci_id, threshold: float) -> Path:
    path = _dataset_results_dir(uci_id) / f"th_{threshold}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_uci_experiment(uci_id, thresholds=settings.THRESHOLDS) -> list[dict]:
    """
    Treina o modelo e monta o `support` UMA ÚNICA VEZ para este `uci_id`
    (accuracy/precision/recall/F1 não dependem do threshold do LIMA).
    Em seguida, para cada valor em `thresholds`, roda o LIMA de novo só
    para gerar as DCs correspondentes -- só isso muda entre as rodadas.

    Retorna uma lista de dicts, um por threshold testado.
    """
    df, target = load_dataset(uci_id)
    x_raw, x_encoded, y = prepare_features(df, target)
    x_train_raw, x_train_encoded, x_test_encoded, y_train, y_test = split_dataset(
        x_raw, x_encoded, y
    )
    clf = train_model(x_train_encoded, y_train)

    section("ETAPA 5: GERANDO SUPORTE DE FILTRO DE DCs")
    y_pred = clf.predict(x_test_encoded)
    support = clf.get_dc_filter_support(
        x_test_encoded,
        predictions=y_pred,
        only_trees_voting_for_prediction=True,
        inbag_only=True,
    )

    metrics = compute_metrics(y_test, y_pred)

    base_result = {
        "uci_id": uci_id,
        "target_column": target,
        "n_train": int(len(x_train_encoded)),
        "n_test": int(len(x_test_encoded)),
        **metrics,
    }

    if not support:
        print("\n  Nada para salvar (nenhuma predição no recorte analisado).")
        result = {**base_result, "threshold": None, "n_dcs": 0, "denial_constraints": []}

        results_path = _dataset_results_dir(uci_id) / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return [result]

    all_results: list[dict] = []

    for threshold in thresholds:
        section(f"ETAPA 6: GERANDO DCs COM O LIMA (JAVA) -- threshold={threshold}")
        global_dcs = generate_dcs_for_support(
            support, x_train_raw, y_train, threshold=threshold
        )

        section("ETAPA 7: SALVANDO RESULTADO")
        threshold_dir = _threshold_results_dir(uci_id, threshold)

        support_path = threshold_dir / "dc_filter_support.json"
        with open(support_path, "w", encoding="utf-8") as f:
            json.dump(support, f, indent=2, ensure_ascii=False)
        print(f"  Salvo em '{support_path}' ({len(global_dcs)} DCs globais)")

        result = {
            **base_result,
            "threshold": threshold,
            "n_dcs": len(global_dcs),
            "denial_constraints": global_dcs,
        }

        results_path = threshold_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  Métricas e DCs salvas em '{results_path}'")

        all_results.append(result)

    return all_results
