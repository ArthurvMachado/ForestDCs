"""
dc_filter.py -- Filtro de tuplas de treino baseado em Denial Constraints.

Usa só DCs com predicados `=` (EQ) e `!=` (NE) -- na prática, dependências
funcionais. Etapa 6.7 do pipeline.

Principais componentes
----------------------
is_eq_ne_only(dc) -> bool
    True se a DC tem ao menos um predicado e todos são EQ ou NE.

DCViolationFilter(x_train_raw, dcs, max_violations=0)
    No __init__, valida cada DC e descarta (guardando o motivo em
    `.discarded`) as que: (a) têm predicado não reconhecido pelo regex,
    (b) usam operador além de =/!=, ou (c) citam colunas ausentes de
    x_train_raw (ex.: o target). Marca também se a DC é "simétrica" (todo
    predicado compara a mesma coluna dos dois lados), para checar uma só
    orientação.
      _index(col)            índice invertido valor -> posições (lazy).
      _violators(dc, q, swap) posições de treino que violam a DC com `q`.
      removed_positions(q)   posições a REMOVER para a instância `q`
                             (com cache por valores das colunas usadas).
      removal_fn(x_test_raw) devolve `fn(posição_da_linha_no_teste) ->
                             posições a remover`, o formato consumido por
                             `TraceableRandomForestClassifier.predict_dc_filtered`.
      n_dcs_used / columns   propriedades para log.

Ideia: dada a instância `q` sendo classificada, descobrir quais tuplas `r` do
treino formam com ela um par que VIOLA alguma DC. Essas tuplas são removidas
das folhas da Random Forest antes da agregação dos votos.

Exemplo (cep -> cidade):

    DC: !(t1.cep = t2.cep & t1.cidade != t2.cidade)

    q = (cep=100, cidade=CM)
    -> índice invertido de `cep`:  100 -> {posições de treino com cep=100}
    -> candidatas = índice["cep"][100]            (lookup O(1), sem varrer a base)
    -> viola = candidatas com cidade != "CM"      (checagem só nas candidatas)

Em vez de comparar `q` com a base inteira (loop O(n) por DC por instância),
o hash restringe a comparação às tuplas que já satisfazem os predicados `=`.
Os predicados `!=` são aplicados depois, vetorizados, só sobre as candidatas.
(DCs só com `!=` não têm o que indexar e caem numa varredura O(n).)

`max_violations` é a tolerância: a tupla só é removida se violar MAIS que
esse número de DCs (cada DC conta no máximo 1x). Com 0, qualquer violação remove.

Convenção de posições: todas as posições devolvidas são posições (0..n-1)
na ordem das linhas de `x_train_raw`, que é a mesma ordem usada no `fit`
da floresta (x_train_encoded).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .dc_conv import DenialConstraint, parse_dc

_EMPTY = np.array([], dtype=np.intp)


def is_eq_ne_only(dc: DenialConstraint) -> bool:
    """True se a DC tem ao menos um predicado e todos são EQ ou NE."""
    return bool(dc.predicates) and all(p.op in ("EQ", "NE") for p in dc.predicates)


class DCViolationFilter:
    """
    Encontra, via índice invertido, as tuplas de treino que violam alguma DC
    (apenas DCs com `=`/`!=`) quando pareadas com uma instância `q`.

    Parâmetros
    ----------
    x_train_raw : DataFrame com os valores ORIGINAIS (não codificados) do
        treino, os mesmos que foram dados ao LIMA.
    dcs : lista de strings no formato de saída do LIMA.
    max_violations : tolerância. Uma tupla de treino só é removida se violar
        MAIS que `max_violations` DCs (contando cada DC no máximo uma vez)
        junto com `q`. Com 0 (padrão), qualquer violação remove a tupla.
    """

    def __init__(
        self,
        x_train_raw: pd.DataFrame,
        dcs: list[str],
        max_violations: int = 0,
    ):
        if max_violations < 0:
            raise ValueError("max_violations deve ser >= 0.")

        self.max_violations = max_violations
        self.n_train = len(x_train_raw)

        # (DC, simétrica?). Simétrica = trocar t1 <-> t2 não muda nada, o que
        # vale quando todo predicado compara a mesma coluna dos dois lados
        # (caso típico de FD). Aí só precisamos checar uma orientação.
        self._dcs: list[tuple[DenialConstraint, bool]] = []
        self.discarded: list[tuple[str, str]] = []  # (dc, motivo)

        for dc_str in dcs:
            dc = parse_dc(dc_str)

            # Segurança: se algum predicado não casou com o regex (ex.: constante),
            # a DC ficaria "mais larga" do que é de verdade. Melhor descartar.
            # O LIMA escreve "p1 & p2 & )" (há um "&" sobrando antes do ")"), então
            # contar "&" + 1 superestima o nº de predicados. Contamos só os
            # trechos não vazios entre os "&".
            n_expected = sum(1 for part in dc_str.strip()[2:-1].split("&") if part.strip())
            if len(dc.predicates) != n_expected:
                self.discarded.append((dc_str, "predicado não reconhecido"))
                continue
            if not is_eq_ne_only(dc):
                self.discarded.append((dc_str, "tem operador além de = e !="))
                continue

            missing = {
                c
                for p in dc.predicates
                for c in (p.col_left, p.col_right)
                if c not in x_train_raw.columns
            }
            if missing:
                self.discarded.append((dc_str, f"colunas ausentes: {sorted(missing)}"))
                continue

            symmetric = all(p.col_left == p.col_right for p in dc.predicates)
            self._dcs.append((dc, symmetric))

        # Colunas que aparecem em alguma DC usada (vão para a chave de cache
        # e para o dicionário `q`).
        self._columns: list[str] = sorted(
            {
                c
                for dc, _ in self._dcs
                for p in dc.predicates
                for c in (p.col_left, p.col_right)
            }
        )
        self._values: dict[str, np.ndarray] = {
            c: x_train_raw[c].to_numpy() for c in self._columns
        }
        self._indexes: dict[str, dict] = {}  # coluna -> {valor: posições}, sob demanda
        self._cache: dict[tuple, np.ndarray] = {}

    # ── propriedades / log ─────────────────────────────────────
    @property
    def n_dcs_used(self) -> int:
        return len(self._dcs)

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    # ── índice invertido ───────────────────────────────────────
    def _index(self, col: str) -> dict:
        """Índice invertido `valor -> array de posições` de uma coluna.
        Construído uma vez, na primeira vez que a coluna é usada num `=`.
        NaN fica de fora (NaN = NaN é falso, igual ao numpy no dc_checker)."""
        idx = self._indexes.get(col)
        if idx is None:
            values = self._values[col]
            idx = pd.Series(values).groupby(values).indices
            self._indexes[col] = idx
        return idx

    # ── núcleo: quem viola uma DC com q ────────────────────────
    def _violators(self, dc: DenialConstraint, q: dict, swap: bool) -> np.ndarray:
        """Posições de treino `r` tais que o par viola `dc`.

        swap=False: q = t1, r = t2  (r usa col_right, q usa col_left)
        swap=True : q = t2, r = t1  (r usa col_left,  q usa col_right)
        """
        eq_lists: list[np.ndarray] = []
        ne_checks: list[tuple[str, object]] = []

        for p in dc.predicates:
            r_col, q_col = (p.col_left, p.col_right) if swap else (p.col_right, p.col_left)
            q_val = q[q_col]
            if p.op == "EQ":
                rows = self._index(r_col).get(q_val)
                if rows is None:
                    return _EMPTY  # ninguém tem esse valor -> nenhum par viola
                eq_lists.append(rows)
            else:  # NE
                ne_checks.append((r_col, q_val))

        if eq_lists:
            # Parte da menor lista de candidatas e interseciona com as demais.
            eq_lists.sort(key=len)
            cand = eq_lists[0]
            for other in eq_lists[1:]:
                cand = np.intersect1d(cand, other, assume_unique=True)
                if not len(cand):
                    return _EMPTY
        else:
            # DC só com `!=`: não há o que indexar, cai para varredura (O(n)).
            cand = np.arange(self.n_train, dtype=np.intp)

        for r_col, q_val in ne_checks:
            if not len(cand):
                break
            cand = cand[self._values[r_col][cand] != q_val]

        return cand

    def removed_positions(self, q: dict) -> np.ndarray:
        """
        Posições de treino a REMOVER para a instância `q` (dict coluna -> valor
        raw, contendo ao menos `self.columns`).

        Uma tupla é removida se violar mais que `max_violations` DCs com `q`.
        """
        key = tuple(q[c] for c in self._columns)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        hits: list[np.ndarray] = []
        for dc, symmetric in self._dcs:
            v = self._violators(dc, q, swap=False)
            if not symmetric:
                v2 = self._violators(dc, q, swap=True)
                if len(v2):
                    v = np.union1d(v, v2)  # cada DC conta no máx. 1x por tupla
            if len(v):
                hits.append(v)

        if not hits:
            result = _EMPTY
        else:
            all_pos = np.concatenate(hits)
            if self.max_violations == 0:
                result = np.unique(all_pos)
            else:
                pos, counts = np.unique(all_pos, return_counts=True)
                result = pos[counts > self.max_violations]

        self._cache[key] = result
        return result

    def removal_fn(self, x_test_raw: pd.DataFrame) -> Callable[[int], np.ndarray]:
        """Devolve `fn(sample_pos) -> posições a remover`, onde sample_pos é
        a posição da linha em `x_test_raw` (mesma ordem de x_test_encoded)."""
        arrays = {c: x_test_raw[c].to_numpy() for c in self._columns}
        cols = self._columns

        def fn(sample_pos: int) -> np.ndarray:
            return self.removed_positions({c: arrays[c][sample_pos] for c in cols})

        return fn