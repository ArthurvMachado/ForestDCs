"""
Testes básicos de sanidade para TraceableRandomForestClassifier, usando o
Iris (via sklearn, sem depender de rede) como dataset de brinquedo.

Rodar com: pytest
(requer `pythonpath = src` no pytest.ini, já incluso neste projeto)
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from dcforest.forest import TraceableRandomForestClassifier


@pytest.fixture
def fitted_clf():
    X, y = load_iris(return_X_y=True, as_frame=True)
    # Problema binário simples: classe 0 vs. resto.
    y = (y == 0).astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = TraceableRandomForestClassifier(
        n_estimators=10, max_depth=3, random_state=42, bootstrap=True
    )
    clf.fit(x_train, y_train)
    return clf, x_test, y_test


def test_predict_matches_sklearn_shape(fitted_clf):
    clf, x_test, y_test = fitted_clf
    y_pred = clf.predict(x_test)
    assert y_pred.shape == (len(x_test),)
    assert set(np.unique(y_pred)).issubset({0, 1})


def test_dc_filter_support_structure(fitted_clf):
    clf, x_test, y_test = fitted_clf
    y_pred = clf.predict(x_test)

    support = clf.get_dc_filter_support(
        x_test,
        predictions=y_pred,
        only_trees_voting_for_prediction=True,
        inbag_only=True,
    )

    assert len(support) == len(x_test)

    for sample in support:
        assert "sample_pos" in sample
        assert "prediction" in sample
        assert "groups" in sample

        for group in sample["groups"]:
            assert group["tree_vote"] == sample["prediction"]
            assert "candidate_columns" in group
            assert "train_ids_in_leaf" in group
            assert group["n_in_leaf"] == len(group["train_ids_in_leaf"])


def test_only_trees_voting_for_prediction_filters_groups(fitted_clf):
    clf, x_test, _ = fitted_clf
    y_pred = clf.predict(x_test)

    support_all_trees = clf.get_dc_filter_support(
        x_test,
        predictions=y_pred,
        only_trees_voting_for_prediction=False,
        inbag_only=True,
    )
    support_only_voting = clf.get_dc_filter_support(
        x_test,
        predictions=y_pred,
        only_trees_voting_for_prediction=True,
        inbag_only=True,
    )

    # Filtrar só árvores que votaram na predição final nunca deve gerar
    # mais grupos do que considerar todas as árvores.
    for sample_all, sample_voting in zip(support_all_trees, support_only_voting):
        assert len(sample_voting["groups"]) <= len(sample_all["groups"])
