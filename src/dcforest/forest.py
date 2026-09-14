"""
Random Forest "rastreável": uma subclasse de RandomForestClassifier que
expõe, para cada amostra prevista, quais folhas de cada árvore foram
atingidas e quais registros de treino caem nelas.

Esse suporte (`get_dc_filter_support`) é o que alimenta a descoberta de
Denial Constraints em `dcforest.dc_discovery`, escopando as DCs às colunas
tocadas no caminho de decisão até cada folha.
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted


class TraceableRandomForestClassifier(RandomForestClassifier):
    """
    RandomForestClassifier that can report, for each test sample:

    - forest prediction
    - tree-level votes
    - leaf reached in each tree
    - training IDs inside each reached leaf
    - decision path used by each tree
    - features touched along each path
    """

    def fit(
        self,
        X: Any,
        y: Sequence,
        sample_weight: Sequence | None = None,
        *,
        train_ids: Sequence | None = None,
    ):
        super().fit(X, y, sample_weight=sample_weight)

        n_train = len(X)

        if train_ids is None:
            if hasattr(X, "index"):
                train_ids = np.asarray(X.index)
            else:
                train_ids = np.arange(n_train)

        train_ids = np.asarray(train_ids)

        if len(train_ids) != n_train:
            raise ValueError(
                f"`train_ids` must have length {n_train}, got {len(train_ids)}."
            )

        self._support_train_ids_ = train_ids
        self._support_y_train_ = np.asarray(y)

        # Shape: (n_train, n_estimators)
        self._support_train_leaves_ = self.apply(X)

        return self

    def predict(
        self,
        X: Any,
        *,
        return_support: bool = False,
        print_support: bool = False,
        only_trees_voting_for_prediction: bool = True,
        inbag_only: bool = True,
        only_records_with_tree_vote: bool = False,
        max_ids_per_leaf: int | None = 20,
        show_path: bool = True,
        show_ids: bool = True,
    ):
        y_pred = super().predict(X)

        if not return_support and not print_support:
            return y_pred

        support = self.get_prediction_support(
            X,
            predictions=y_pred,
            only_trees_voting_for_prediction=only_trees_voting_for_prediction,
            inbag_only=inbag_only,
            only_records_with_tree_vote=only_records_with_tree_vote,
            max_ids_per_leaf=max_ids_per_leaf,
        )

        if print_support:
            self.print_prediction_support(
                support,
                show_path=show_path,
                show_ids=show_ids,
            )

        if return_support:
            return y_pred, support

        return y_pred

    @staticmethod
    def _python_scalar(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _leaf_gini(tree, leaf_id: int) -> float:
        """
        Impureza de Gini do nó `leaf_id`, conforme calculada pelo próprio
        scikit-learn durante o fit (tree_.impurity), assumindo criterion="gini"
        (default do RandomForestClassifier). 0.0 = folha pura.
        """
        return float(tree.tree_.impurity[leaf_id])

    def _as_tree_input(self, X: Any) -> np.ndarray:
        if hasattr(X, "to_numpy"):
            return X.to_numpy()
        return np.asarray(X)

    def _feature_names(self, X: Any, X_for_trees: np.ndarray) -> list[str]:
        if hasattr(X, "columns"):
            return [str(c) for c in X.columns]

        if hasattr(self, "feature_names_in_"):
            return [str(c) for c in self.feature_names_in_]

        return [f"x{j}" for j in range(X_for_trees.shape[1])]

    def _extract_tree_path(
        self,
        tree,
        X_for_trees: np.ndarray,
        sample_pos: int,
        feature_names: list[str],
    ) -> dict[str, Any]:
        """
        Extract the actual path followed by one sample in one tree.
        """
        tree_struct = tree.tree_

        sample_2d = X_for_trees[sample_pos : sample_pos + 1]

        node_indicator = tree.decision_path(sample_2d)
        node_ids = node_indicator.indices[
            node_indicator.indptr[0] : node_indicator.indptr[1]
        ]

        path = []

        for step, node_id in enumerate(node_ids[:-1]):
            node_id = int(node_id)

            feature_idx = int(tree_struct.feature[node_id])
            threshold = float(tree_struct.threshold[node_id])

            next_node_id = int(node_ids[step + 1])

            left_child = int(tree_struct.children_left[node_id])
            right_child = int(tree_struct.children_right[node_id])

            if next_node_id == left_child:
                operator = "<="
            elif next_node_id == right_child:
                operator = ">"
            else:
                operator = "?"

            raw_value = X_for_trees[sample_pos, feature_idx]
            value = self._python_scalar(raw_value)

            path.append(
                {
                    "node_id": node_id,
                    "feature_idx": feature_idx,
                    "feature_name": feature_names[feature_idx],
                    "threshold": threshold,
                    "sample_value": value,
                    "operator_taken": operator,
                    "condition_taken": (
                        f"{feature_names[feature_idx]} "
                        f"{operator} "
                        f"{threshold:.6g}"
                    ),
                }
            )

        return {
            "leaf_id": int(node_ids[-1]),
            "path": path,
            "features_touched": sorted(
                {step["feature_name"] for step in path}
            ),
        }

    def get_prediction_support(
        self,
        X: Any,
        *,
        predictions: Sequence | None = None,
        only_trees_voting_for_prediction: bool = True,
        inbag_only: bool = True,
        only_records_with_tree_vote: bool = False,
        max_ids_per_leaf: int | None = 20,
    ) -> list[dict[str, Any]]:
        check_is_fitted(
            self,
            attributes=[
                "estimators_",
                "_support_train_ids_",
                "_support_y_train_",
                "_support_train_leaves_",
            ],
        )

        if predictions is None:
            predictions = super().predict(X)

        predictions = np.asarray(predictions)

        # Shape: (n_test, n_estimators)
        test_leaves = self.apply(X)

        X_for_trees = self._as_tree_input(X)
        feature_names = self._feature_names(X, X_for_trees)

        # Hard vote of each individual tree.
        # Shape: (n_test, n_estimators)
        tree_votes = np.column_stack(
            [tree.predict(X_for_trees) for tree in self.estimators_]
        )

        if inbag_only:
            if not hasattr(self, "estimators_samples_"):
                raise RuntimeError(
                    "`inbag_only=True` requires `estimators_samples_`, "
                    "which is unavailable in this sklearn version."
                )

            estimators_samples = self.estimators_samples_
        else:
            estimators_samples = None

        # Pré-computa, uma vez por árvore, leaf_id -> posições de treino
        # naquela folha (ver explicação equivalente em get_dc_filter_support).
        tree_leaf_index: dict[int, dict[int, np.ndarray]] = {}
        for tree_id in range(len(self.estimators_)):
            if inbag_only:
                candidate_positions_full = np.asarray(estimators_samples[tree_id])
            else:
                candidate_positions_full = np.arange(len(self._support_train_ids_))

            candidate_leaf_ids_full = self._support_train_leaves_[
                candidate_positions_full,
                tree_id,
            ]

            order = np.argsort(candidate_leaf_ids_full, kind="stable")
            sorted_leaf_ids = candidate_leaf_ids_full[order]
            sorted_positions = candidate_positions_full[order]

            unique_leaves, start_idx = np.unique(sorted_leaf_ids, return_index=True)
            split_positions = np.split(sorted_positions, start_idx[1:])

            tree_leaf_index[tree_id] = dict(zip(unique_leaves.tolist(), split_positions))

        empty_positions = np.array([], dtype=np.int64)

        all_support: list[dict[str, Any]] = []

        for sample_pos in range(test_leaves.shape[0]):
            prediction = self._python_scalar(predictions[sample_pos])

            sample_support = {
                "sample_pos": sample_pos,
                "prediction": prediction,
                "trees": [],
            }

            for tree_id, leaf_id in enumerate(test_leaves[sample_pos]):
                tree = self.estimators_[tree_id]

                leaf_id = int(leaf_id)
                tree_vote = self._python_scalar(tree_votes[sample_pos, tree_id])

                if only_trees_voting_for_prediction and tree_vote != prediction:
                    continue

                positions_in_leaf = tree_leaf_index[tree_id].get(
                    leaf_id, empty_positions
                )

                if only_records_with_tree_vote:
                    same_label_mask = (
                        self._support_y_train_[positions_in_leaf] == tree_vote
                    )
                    positions_in_leaf = positions_in_leaf[same_label_mask]

                ids_in_leaf = self._support_train_ids_[positions_in_leaf]
                labels_in_leaf = self._support_y_train_[positions_in_leaf]

                n_total = len(ids_in_leaf)

                if max_ids_per_leaf is not None:
                    shown_ids = ids_in_leaf[:max_ids_per_leaf]
                    shown_labels = labels_in_leaf[:max_ids_per_leaf]
                else:
                    shown_ids = ids_in_leaf
                    shown_labels = labels_in_leaf

                # This was missing in your version.
                path_info = self._extract_tree_path(
                    tree=tree,
                    X_for_trees=X_for_trees,
                    sample_pos=sample_pos,
                    feature_names=feature_names,
                )

                sample_support["trees"].append(
                    {
                        "tree_id": tree_id,
                        "leaf_id": leaf_id,
                        "tree_vote": tree_vote,
                        "n_training_records_in_leaf": n_total,
                        "train_ids_in_leaf": shown_ids.tolist(),
                        "train_labels_in_leaf": shown_labels.tolist(),
                        "truncated": (
                            max_ids_per_leaf is not None
                            and n_total > max_ids_per_leaf
                        ),
                        "path": path_info["path"],
                        "features_touched": path_info["features_touched"],
                    }
                )

            all_support.append(sample_support)

        return all_support

    def get_dc_filter_support(
        self,
        X: Any,
        *,
        predictions: Sequence | None = None,
        only_trees_voting_for_prediction: bool = True,
        inbag_only: bool = True,
        only_records_with_tree_vote: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Versão reduzida de `get_prediction_support`, pensada para alimentar
        um algoritmo de descoberta de Denial Constraints (DCs).

        Para cada amostra de teste, devolve apenas:

          - prediction da forest
          - lista de "groups" (um por árvore que votou na predição), cada um com:
              - tree_id, leaf_id
              - tree_vote
              - leaf_gini       -> impureza de Gini da folha (0 = folha pura)
              - n_in_leaf       -> total de registros de treino na folha
                                    (sem truncamento; é exatamente o grupo
                                    a ser usado como filtro/escopo das DCs)
              - train_ids_in_leaf -> TODOS os IDs de treino na folha
              - candidate_columns -> features tocadas no caminho até a folha
                                      (escopo de colunas candidatas para a DC)

        Não inclui: thresholds/condições nó-a-nó, labels de treino na folha,
        nem truncamento de IDs — tudo isso é dispensável para o caso de uso
        de filtro/escopo, e omiti-lo deixa o arquivo bem mais compacto.
        """
        check_is_fitted(
            self,
            attributes=[
                "estimators_",
                "_support_train_ids_",
                "_support_y_train_",
                "_support_train_leaves_",
            ],
        )

        if predictions is None:
            predictions = super().predict(X)

        predictions = np.asarray(predictions)

        # Shape: (n_test, n_estimators)
        test_leaves = self.apply(X)

        X_for_trees = self._as_tree_input(X)
        feature_names = self._feature_names(X, X_for_trees)

        # Hard vote de cada árvore individual.
        tree_votes = np.column_stack(
            [tree.predict(X_for_trees) for tree in self.estimators_]
        )

        if inbag_only:
            if not hasattr(self, "estimators_samples_"):
                raise RuntimeError(
                    "`inbag_only=True` requires `estimators_samples_`, "
                    "which is unavailable in this sklearn version."
                )
            estimators_samples = self.estimators_samples_
        else:
            estimators_samples = None

        # ── Pré-computa, UMA VEZ por árvore, um índice leaf_id -> posições
        # de treino naquela folha. Isso evita recriar (para cada uma das
        # ~n_test x n_estimators combinações) um array do tamanho do
        # bootstrap inteiro só para descobrir quem cai em cada folha —
        # que era o que causava tanto a lentidão quanto o
        # `ArrayMemoryError` (fragmentação de heap por excesso de
        # alocações temporárias repetidas).
        tree_leaf_index: dict[int, dict[int, np.ndarray]] = {}
        for tree_id in range(len(self.estimators_)):
            if inbag_only:
                candidate_positions_full = np.asarray(estimators_samples[tree_id])
            else:
                candidate_positions_full = np.arange(len(self._support_train_ids_))

            candidate_leaf_ids_full = self._support_train_leaves_[
                candidate_positions_full,
                tree_id,
            ]

            order = np.argsort(candidate_leaf_ids_full, kind="stable")
            sorted_leaf_ids = candidate_leaf_ids_full[order]
            sorted_positions = candidate_positions_full[order]

            unique_leaves, start_idx = np.unique(sorted_leaf_ids, return_index=True)
            split_positions = np.split(sorted_positions, start_idx[1:])

            tree_leaf_index[tree_id] = dict(zip(unique_leaves.tolist(), split_positions))

        empty_positions = np.array([], dtype=np.int64)

        # ── Cache por (tree_id, leaf_id): tree_vote, leaf_gini, n_in_leaf,
        # train_ids_in_leaf e candidate_columns dependem só da folha, não
        # da amostra de teste específica. Sem esse cache, cada amostra que
        # cai numa folha já vista recria do zero (inclusive `.tolist()`
        # sobre milhares de IDs) o mesmo conteúdo — com ~100k amostras e
        # só ~256 folhas por árvore, isso significa guardar o mesmo grupo
        # centenas de vezes na memória, o que causava o `MemoryError`.
        leaf_group_cache: dict[tuple[int, int], dict[str, Any]] = {}

        all_support: list[dict[str, Any]] = []

        for sample_pos in range(test_leaves.shape[0]):
            prediction = self._python_scalar(predictions[sample_pos])

            sample_support: dict[str, Any] = {
                "sample_pos": sample_pos,
                "prediction": prediction,
                "groups": [],
            }

            for tree_id, leaf_id in enumerate(test_leaves[sample_pos]):
                tree = self.estimators_[tree_id]

                leaf_id = int(leaf_id)
                tree_vote = self._python_scalar(tree_votes[sample_pos, tree_id])

                if only_trees_voting_for_prediction and tree_vote != prediction:
                    continue

                cache_key = (tree_id, leaf_id)
                cached_group = leaf_group_cache.get(cache_key)

                if cached_group is None:
                    positions_in_leaf = tree_leaf_index[tree_id].get(
                        leaf_id, empty_positions
                    )

                    if only_records_with_tree_vote:
                        same_label_mask = (
                            self._support_y_train_[positions_in_leaf] == tree_vote
                        )
                        positions_in_leaf = positions_in_leaf[same_label_mask]

                    ids_in_leaf = self._support_train_ids_[positions_in_leaf]

                    path_info = self._extract_tree_path(
                        tree=tree,
                        X_for_trees=X_for_trees,
                        sample_pos=sample_pos,
                        feature_names=feature_names,
                    )

                    cached_group = {
                        "tree_id": tree_id,
                        "leaf_id": leaf_id,
                        "tree_vote": tree_vote,
                        "leaf_gini": self._leaf_gini(tree, leaf_id),
                        "n_in_leaf": int(len(ids_in_leaf)),
                        "train_ids_in_leaf": ids_in_leaf.tolist(),
                        "candidate_columns": path_info["features_touched"],
                    }
                    leaf_group_cache[cache_key] = cached_group

                sample_support["groups"].append(cached_group)

            all_support.append(sample_support)

        return all_support

    @staticmethod
    def print_prediction_support(
        support: list[dict[str, Any]],
        *,
        show_path: bool = True,
        show_ids: bool = True,
    ) -> None:
        for sample in support:
            print(
                f"\nTest sample position {sample['sample_pos']} "
                f"-> forest prediction: {sample['prediction']}"
            )

            for tree_info in sample["trees"]:
                suffix = " ..." if tree_info["truncated"] else ""

                print(
                    f"  tree={tree_info['tree_id']:03d} "
                    f"leaf={tree_info['leaf_id']} "
                    f"tree_vote={tree_info['tree_vote']} "
                    f"n_leaf={tree_info['n_training_records_in_leaf']}"
                )

                if show_path:
                    print(
                        f"    features touched: "
                        f"{tree_info['features_touched']}"
                    )

                    for step in tree_info["path"]:
                        value = step["sample_value"]

                        try:
                            value_str = f"{float(value):.6g}"
                        except (TypeError, ValueError):
                            value_str = str(value)

                        print(
                            f"    node={step['node_id']} "
                            f"{step['feature_name']} "
                            f"{step['operator_taken']} "
                            f"{step['threshold']:.6g} "
                            f"(value={value_str})"
                        )

                if show_ids:
                    print(
                        f"    ids={tree_info['train_ids_in_leaf']}{suffix}"
                    )