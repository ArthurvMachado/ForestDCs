"""
dc_conv.py -- Representação e verificação de Denial Constraints (DCs).

Lê as DCs no formato de saída do LIMA e confere, direto no DataFrame, quais
pares de linhas as violam. É a verificação "no dataset todo" (ETAPA 6.5).

Formato aceito (uma DC por string):

    !(Coluna(Tipo) OP Coluna(Tipo) & Coluna(Tipo) OP Coluna(Tipo) & ...)

com OP em {EQ, NE, LT, LE, GT, GE}. Uma DC diz que NÃO pode existir um par de
tuplas (t1, t2), com t1 != t2, em que TODOS os predicados sejam verdadeiros
ao mesmo tempo; se existir, o par "viola" a DC.

Principais componentes
----------------------
Predicate
    Um predicado `col_left OP col_right`. `matrix(df)` devolve a matriz
    booleana n x n com M[i, j] = df[col_left][i] OP df[col_right][j]
    (linha i faz o papel de t1, linha j o de t2).

DenialConstraint
    Lista de predicados + a string original (`raw`).
    - violation_matrix(df): AND de todas as matrizes dos predicados, com a
      diagonal zerada (t1 != t2). Custo O(n^2) em tempo e memória.
    - violating_pairs(df): converte a matriz em pares de rótulos de índice
      (tipos nativos do Python, para poder ir para JSON).

parse_dc(dc_str) -> DenialConstraint
    Faz o parsing por regex. Predicados que não casam com o padrão (ex.:
    comparação com constante) são ignorados em silêncio; quem precisa de
    segurança contra isso confere a contagem (ver dc_filter.py).

check_all_dcs(df, dcs, max_pairs_shown=20) -> DataFrame
    Uma linha por DC: `dc`, `predicados`, `n_pares_violando` e
    `exemplos_pares` (limitado a `max_pairs_shown`).
    Atenção: as colunas citadas na DC precisam existir em `df`. Se o LIMA
    tiver recebido o target como coluna (settings.TARGET_COLUMN), DCs que o
    citam não poderão ser checadas em x_train_raw/x_test_raw.

Uso em linha de comando (bloco __main__):
    python dc_checker.py resultados.json dataset.csv
onde o JSON tem a chave "denial_constraints" e o CSV tem os nomes de coluna
SEM a anotação de tipo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Extrai "Coluna(Tipo) OP Coluna(Tipo)" de dentro do "!( ... )"
_PRED_RE = re.compile(
    r"(\w+)\(\w+\)\s*(LE|LT|GT|GE|EQ|NE)\s*(\w+)\(\w+\)"
)

_OPS = {
    "EQ": np.equal,
    "NE": np.not_equal,
    "LT": np.less,
    "LE": np.less_equal,
    "GT": np.greater,
    "GE": np.greater_equal,
}


@dataclass
class Predicate:
    col_left: str
    op: str
    col_right: str

    def matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Matriz booleana nxn: M[i, j] = (df[col_left][i] OP df[col_right][j])."""
        a = df[self.col_left].to_numpy()[:, None]   # coluna -> t1 (linhas)
        b = df[self.col_right].to_numpy()[None, :]  # coluna -> t2 (colunas)
        return _OPS[self.op](a, b)

    def __str__(self) -> str:
        return f"{self.col_left} {self.op} {self.col_right}"


@dataclass
class DenialConstraint:
    raw: str
    predicates: list[Predicate] = field(default_factory=list)

    def violation_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """True em [i, j] quando o PAR (linha i, linha j) viola a DC, ou
        seja, quando todos os predicados valem ao mesmo tempo."""
        n = len(df)
        result = np.ones((n, n), dtype=bool)
        for pred in self.predicates:
            result &= pred.matrix(df)
        np.fill_diagonal(result, False)  # t1 != t2
        return result

    def violating_pairs(self, df: pd.DataFrame) -> list[tuple]:
        matrix = self.violation_matrix(df)
        idx_i, idx_j = np.where(matrix)
        labels_i = df.index[idx_i]
        labels_j = df.index[idx_j]
        return [
            (self._to_native(a), self._to_native(b))
            for a, b in zip(labels_i, labels_j)
        ]

    @staticmethod
    def _to_native(value):
        """Converte numpy.int64/np.generic para tipo Python nativo, para
        não estourar json.dump depois (mesmo padrão de `_python_scalar`
        em forest.py)."""
        if isinstance(value, np.generic):
            return value.item()
        return value


def parse_dc(dc_str: str) -> DenialConstraint:
    """Faz o parsing de uma linha de output do LIMA em uma DenialConstraint."""
    inner = dc_str.strip()
    if inner.startswith("!(") and inner.endswith(")"):
        inner = inner[2:-1]
    preds = [
        Predicate(col_left=m.group(1), op=m.group(2), col_right=m.group(3))
        for m in _PRED_RE.finditer(inner)
    ]
    return DenialConstraint(raw=dc_str, predicates=preds)


def check_all_dcs(df: pd.DataFrame, dcs: list[str], max_pairs_shown: int = 20) -> pd.DataFrame:
    """Para cada DC, conta quantos pares de linhas violam e devolve um
    resumo tabular (uma linha por DC)."""
    rows = []
    for dc_str in dcs:
        dc = parse_dc(dc_str)
        pairs = dc.violating_pairs(df)
        rows.append({
            "dc": dc_str,
            "predicados": " & ".join(str(p) for p in dc.predicates),
            "n_pares_violando": len(pairs),
            "exemplos_pares": pairs[:max_pairs_shown],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import json
    import sys

    # Exemplo: python dc_checker.py resultados.json dataset.csv
    with open(sys.argv[1], encoding="utf-8") as f:
        result = json.load(f)
    dcs = result["denial_constraints"]

    df = pd.read_csv(sys.argv[2])
    resumo = check_all_dcs(df, dcs)
    pd.set_option("display.max_colwidth", 60)
    print(resumo[["n_pares_violando", "predicados"]])