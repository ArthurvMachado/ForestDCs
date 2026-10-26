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

import numpy as np

from configs import settings
from .data_loading import load_dataset, prepare_features, section, split_dataset
from .dc_conv import check_all_dcs
from .dc_discovery import generate_dcs_for_support
from .dc_filter import DCViolationFilter
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


def run_dc_filtered_prediction(
    clf: TraceableRandomForestClassifier,
    x_train_raw,
    x_test_raw,
    x_test_encoded,
    y_test,
    y_pred_baseline,
    dcs: list[str],
    max_violations_grid=settings.DC_MAX_VIOLATIONS_GRID,
) -> list[dict]:
    """
    Reclassifica o teste removendo, de cada folha atingida, as tuplas de
    treino que violam alguma DC (só `=`/`!=`, via índice invertido) junto
    com a instância classificada. Uma rodada por valor de `max_violations`.
    """
    if not np.array_equal(clf._support_train_ids_, x_train_raw.index.to_numpy()):
        raise ValueError(
            "x_train_raw precisa ter as mesmas linhas, na mesma ordem, do "
            "x_train usado no fit (as posições de treino são compartilhadas)."
        )

    results: list[dict] = []
    for max_violations in max_violations_grid:
        start = time.time()
        dc_filter = DCViolationFilter(x_train_raw, dcs, max_violations=max_violations)
        if max_violations == max_violations_grid[0]:
            print(
                f"  DCs usadas (só = e !=): {dc_filter.n_dcs_used}/{len(dcs)} "
                f"| descartadas: {len(dc_filter.discarded)}"
            )

        y_pred_dc, stats = clf.predict_dc_filtered(
            x_test_encoded,
            dc_filter.removal_fn(x_test_raw),
            empty_leaf_policy=settings.DC_EMPTY_LEAF_POLICY,
            return_stats=True,
        )
        elapsed = time.time() - start

        metrics = compute_metrics(
            y_test,
            y_pred_dc,
            title=f"RESULTADOS COM FILTRO DE DCs -- max_violations={max_violations}",
        )
        n_changed = int((np.asarray(y_pred_dc) != np.asarray(y_pred_baseline)).sum())
        # print(f"  Predições alteradas vs. baseline: {n_changed}/{len(y_pred_dc)}")
        print(f"  Tuplas removidas (média/instância): {stats['n_removed_total'].mean():.1f}")
        print(f"  Folhas esvaziadas (média/instância): {stats['n_trees_empty'].mean():.2f}")
        print(f"  Tempo: {elapsed:.2f}s")

        results.append(
            {
                "max_violations": max_violations,
                "n_dcs_used": dc_filter.n_dcs_used,
                "n_dcs_discarded": len(dc_filter.discarded),
                "n_predictions_changed": n_changed,
                "mean_removed_per_instance": float(stats["n_removed_total"].mean()),
                "mean_empty_trees_per_instance": float(stats["n_trees_empty"].mean()),
                "n_instances_fell_back": int(stats["fell_back"].sum()),
                "seconds": elapsed,
                **metrics,
            }
        )
    return results


def run_uci_experiment(
    uci_id,
    thresholds=settings.THRESHOLDS,
    save_support: bool | None = None,
    check_violations: bool | None = None,
) -> list[dict]:
    """
    Treina o modelo e monta o `support` UMA ÚNICA VEZ para este `uci_id`
    (accuracy/precision/recall/F1 não dependem do threshold do LIMA).
    Em seguida, para cada valor em `thresholds`, roda o LIMA de novo só
    para gerar as DCs correspondentes -- só isso muda entre as rodadas.

    `save_support`: grava `dc_filter_support.json` por threshold (arquivo muito
    grande em datasets grandes). None -> usa `settings.SAVE_SUPPORT`.
    `check_violations`: confere as violações das DCs no treino e no teste
    (ETAPA 6.5, só informativa). None -> usa `settings.CHECK_VIOLATIONS`.

    Retorna uma lista de dicts, um por threshold testado.
    """
    if save_support is None:
        save_support = settings.SAVE_SUPPORT
    if check_violations is None:
        check_violations = settings.CHECK_VIOLATIONS
    df, target = load_dataset(uci_id)
    x_raw, x_encoded, y = prepare_features(df, target)
    x_train_raw, x_train_encoded, x_test_raw, x_test_encoded, y_train, y_test = split_dataset(
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

        if check_violations:
            section(f"ETAPA 6.5: VERIFICANDO VIOLAÇÕES DAS DCs -- threshold={threshold}")
            violacoes_treino = check_all_dcs(x_train_raw, global_dcs)
            violacoes_teste = check_all_dcs(x_test_raw, global_dcs)
            n_viol_treino = int(violacoes_treino["n_pares_violando"].sum())
            n_viol_teste = int(violacoes_teste["n_pares_violando"].sum())
            print(f"  Treino: {n_viol_treino} pares violando (somando as {len(global_dcs)} DCs)")
            print(f"  Teste : {n_viol_teste} pares violando (somando as {len(global_dcs)} DCs)")
        else:
            violacoes_treino = violacoes_teste = None
            n_viol_treino = n_viol_teste = None
            print("\n  Verificação de violações pulada (--skip-violation-check).")

        section(f"ETAPA 6.7: PREDIÇÃO COM FILTRO DE DCs (= e !=) -- threshold={threshold}")
        dc_filtered = run_dc_filtered_prediction(
            clf, x_train_raw, x_test_raw, x_test_encoded, y_test, y_pred, global_dcs
        )

        section("ETAPA 7: SALVANDO RESULTADO")
        threshold_dir = _threshold_results_dir(uci_id, threshold)

        if save_support:
            support_path = threshold_dir / "dc_filter_support.json"
            with open(support_path, "w", encoding="utf-8") as f:
                json.dump(support, f, indent=2, ensure_ascii=False)
            print(f"  Suporte salvo em '{support_path}'")
        else:
            print("  Suporte não salvo (use --save-support para gravar dc_filter_support.json)")
        print(f"  {len(global_dcs)} DCs globais")

        if check_violations:
            violations_path = threshold_dir / "dc_violations.json"
            with open(violations_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "treino": violacoes_treino.to_dict("records"),
                        "teste": violacoes_teste.to_dict("records"),
                    },
                    f, indent=2, ensure_ascii=False,
                )
            print(f"  Violações de DC salvas em '{violations_path}'")

        result = {
            **base_result,
            "threshold": threshold,
            "n_dcs": len(global_dcs),
            "denial_constraints": global_dcs,
            "n_pares_violando_treino": n_viol_treino,
            "n_pares_violando_teste": n_viol_teste,
            "dc_filtered": dc_filtered,
        }

        results_path = threshold_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  Métricas e DCs salvas em '{results_path}'")

        all_results.append(result)

    return all_results