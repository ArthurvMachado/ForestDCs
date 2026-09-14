"""
Wrapper do LIMA (Java) para descoberta de Denial Constraints (DCs).

Roda o LIMA UMA ÚNICA VEZ sobre x_train inteiro, gerando um conjunto
global de DCs. (A função `generate_dcs_for_support`, que antes também
filtrava esse conjunto por folha usando `support`, está com esse trecho
comentado — mesmo comportamento do `generateDCs.py` original.)

Uso típico (dentro de `dcforest.pipeline.run_uci_experiment`):

    from dcforest.dc_discovery import generate_dcs_for_support

    global_dcs = generate_dcs_for_support(support, x_train, y_train, threshold=0.1)
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from configs import settings

# Regex para extrair nomes de coluna de uma linha de DC, ex:
# "!(SingleExemp(Integer) NE SingleExemp(Integer) & MarriedExemp(Integer) GT ...)"
# -> {"SingleExemp", "MarriedExemp", ...}
_COLUMN_PATTERN = re.compile(r"(\w+)\(\w+\)")


def _lima_type(series: pd.Series) -> str:
    """Mapeia o dtype do pandas para a anotação de tipo que o LIMA espera
    no cabeçalho do CSV (ex: 'MaritalStatus(String)', 'Salary(Integer)')."""
    if pd.api.types.is_bool_dtype(series):
        return "String"
    if pd.api.types.is_integer_dtype(series):
        return "Integer"
    if pd.api.types.is_float_dtype(series):
        return "Double"
    return "String"


def _write_typed_csv(df: pd.DataFrame, csv_path: Path) -> None:
    """Escreve o CSV com cabeçalho anotado 'Coluna(Tipo)', formato que o
    LIMA espera (ex: 'MaritalStatus(String),Salary(Integer),Rate(Double)')."""
    typed_header = [f"{col}({_lima_type(df[col])})" for col in df.columns]
    df.to_csv(csv_path, index=False, header=typed_header)


def _extract_columns(dc: str) -> set[str]:
    return set(_COLUMN_PATTERN.findall(dc))


def _run_lima(csv_path: Path, sample_size: int, threshold: float) -> list[str]:
    """Executa o LIMA e lê as DCs do `output.txt` que ele escreve em
    `settings.LIMA_PROJECT_DIR`."""
    comando = [
        "java", "-jar", settings.JAR_RELATIVE_PATH,
        str(csv_path.resolve()), str(threshold), str(sample_size), str(settings.SEED),
    ]

    output_path = settings.LIMA_PROJECT_DIR / settings.OUTPUT_FILENAME

    if output_path.exists():
        output_path.unlink()

    if settings.DEBUG:
        print(f"  [DEBUG] comando: {' '.join(comando)}")
        print(f"  [DEBUG] cwd: {settings.LIMA_PROJECT_DIR}")

    resultado = subprocess.run(
        comando,
        cwd=settings.LIMA_PROJECT_DIR,
        capture_output=True,
        text=True,
        check=True,
        timeout=settings.TIMEOUT_S,
    )

    if settings.DEBUG:
        print("  [DEBUG] --- stdout do Java ---")
        print(resultado.stdout.strip() or "(vazio)")
        print("  [DEBUG] --- stderr do Java ---")
        print(resultado.stderr.strip() or "(vazio)")

    if not output_path.exists():
        raise RuntimeError(
            f"LIMA rodou mas não gerou '{settings.OUTPUT_FILENAME}' em "
            f"{settings.LIMA_PROJECT_DIR}. Veja o stdout/stderr acima "
            f"(settings.DEBUG=True) para pistas."
        )

    dcs = [
        line.strip()
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if settings.DEBUG and not dcs:
        print(f"  [DEBUG] '{settings.OUTPUT_FILENAME}' foi gerado mas está vazio (0 DCs).")

    return dcs


def generate_global_dcs(
    x_train: pd.DataFrame,
    y_train: pd.Series | None = None,
    threshold: float = settings.DEFAULT_THRESHOLD,
) -> list[str]:
    """
    Roda o LIMA uma única vez sobre x_train inteiro (+ y_train, se
    `settings.TARGET_COLUMN` não for None) e retorna a lista de DCs geradas.
    """
    subset = x_train.copy()
    if settings.TARGET_COLUMN is not None and y_train is not None:
        subset[settings.TARGET_COLUMN] = y_train.to_numpy()

    sample_size = settings.SAMPLE_SIZE if settings.SAMPLE_SIZE is not None else len(subset)

    if settings.DEBUG:
        # Mantém o CSV em data/interim/ (não some ao sair da função), pra
        # você poder inspecionar ou rodar o comando java manualmente por fora.
        csv_path = settings.INTERIM_DATA_DIR / "debug_train_full.csv"
        _write_typed_csv(subset, csv_path)
        print(f"  [DEBUG] CSV mantido em: {csv_path}")

        print(f"  Rodando LIMA (threshold={threshold}) sobre x_train inteiro ({len(subset):,} linhas)...")
        dcs = _run_lima(csv_path, sample_size, threshold)
        print(f"  {len(dcs)} DCs geradas.")
        return dcs

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "train_full.csv"
        _write_typed_csv(subset, csv_path)

        print(f"  Rodando LIMA (threshold={threshold}) sobre x_train inteiro ({len(subset):,} linhas)...")
        dcs = _run_lima(csv_path, sample_size, threshold)
        print(f"  {len(dcs)} DCs geradas.")

    return dcs


# def filter_dcs_for_group(global_dcs: list[str], candidate_columns: list[str],
#                           dc_columns_cache: dict[str, set[str]]) -> list[str]:
#     """
#     Mantém apenas as DCs cujas colunas usadas estão inteiramente contidas em
#     `candidate_columns` (as colunas do caminho de decisão até aquela folha).
#     """
#     allowed = set(candidate_columns)
#     return [dc for dc in global_dcs if dc_columns_cache[dc] <= allowed]


def generate_dcs_for_support(
    support: list[dict],
    x_train: pd.DataFrame,
    y_train: pd.Series | None = None,
    threshold: float = settings.DEFAULT_THRESHOLD,
) -> list[str]:
    """
    1. Roda o LIMA uma única vez sobre x_train -> `global_dcs`.
    2. Para cada folha em `support`, filtra `global_dcs` e grava o resultado
       em `group["denial_constraints"]`, in-place (atualmente comentado,
       igual ao comportamento original de `generateDCs.py`).

    Retorna `global_dcs` (útil caso você queira salvar/inspecionar à parte).
    """
    global_dcs = generate_global_dcs(x_train, y_train, threshold=threshold)
    dc_columns_cache = {dc: _extract_columns(dc) for dc in global_dcs}

    total_groups = sum(len(s["groups"]) for s in support)
    done = 0

    # for sample in support:
    #     for group in sample["groups"]:
    #         filtered = filter_dcs_for_group(
    #             global_dcs, group["candidate_columns"], dc_columns_cache
    #         )
    #         group["denial_constraints"] = filtered
    #
    #         done += 1
    #         print(f"  [{done}/{total_groups}] tree={group['tree_id']} "
    #               f"leaf={group['leaf_id']} -> {len(filtered)}/{len(global_dcs)} DCs aplicáveis")

    return global_dcs
