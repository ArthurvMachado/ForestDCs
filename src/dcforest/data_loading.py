from __future__ import annotations
 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo
 
from configs import settings
 
WIDTH = 66
 
 
def section(title: str) -> None:
    """Imprime um cabeçalho de seção no console (log legível do pipeline)."""
    print()
    print("─" * WIDTH)
    print(title.center(WIDTH))
    print("─" * WIDTH)
    print()
 
 
def load_dataset(uci_id) -> tuple[pd.DataFrame, str]:
    """Busca o dataset `uci_id` via ucimlrepo e binariza o target contra
    a primeira classe encontrada. Retorna (df, nome_da_coluna_target)."""
    section("ETAPA 1: CARREGANDO O DATASET")
 
    ds = fetch_ucirepo(id=uci_id)
    target_col = ds.data.targets.columns[0]
    print(target_col)
 
    X = ds.data.features
    y = ds.data.targets
 
    df = pd.concat([X, y], axis=1)
 
    print(df[target_col].unique())
 
    target = df[target_col].unique()[0]
    df[target_col] = (df[target_col] == target).astype(int)
 
    print(df.head(100))
    print(df[target_col].unique())
 
    print(f"  Dataset carregado: {len(df):,} linhas, {len(df.columns)} colunas")
    return df, target_col
 
 
def prepare_features(
    df: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Separa features/target e devolve duas versões de X: `x_raw` (valores
    originais, para o LIMA) e `x_encoded` (colunas categóricas codificadas,
    para o classificador).
 
    As colunas categóricas são detectadas automaticamente pelo dtype
    (texto/objeto ou category), em vez de depender de uma lista fixa em
    `settings.CATEGORICAL_COLUMNS` — datasets diferentes da UCI têm
    colunas categóricas diferentes, então uma lista hardcoded exigiria
    edição manual a cada novo dataset testado.
    """
    section("ETAPA 2: PREPARANDO FEATURES")
 
    df = df.drop(columns=[c for c in settings.DROP_COLUMNS if c in df.columns])
 
    target_val = df[target]
    y = target_val.map({"Y": settings.POSITIVE_LABEL, "N": settings.NEGATIVE_LABEL})
    if y.isna().all():
        y = pd.to_numeric(target_val, errors="raise").astype(int)
    x_raw = df.drop(columns=[target])
 
    x_encoded = x_raw.copy()
    categorical_columns = [
        col
        for col in x_encoded.columns
        if not pd.api.types.is_numeric_dtype(x_encoded[col])
    ]
 
    encoders: dict[str, LabelEncoder] = {}
    for col in categorical_columns:
        le = LabelEncoder()
        x_encoded[col] = le.fit_transform(x_encoded[col])
        encoders[col] = le
 
    print(f"  Features usadas: {list(x_raw.columns)}")
    print(f"  Colunas categóricas codificadas: {categorical_columns or '(nenhuma)'}")
    print(f"  Distribuição do target: 0={sum(y == 0)}  1={sum(y == 1)}")
 
    return x_raw, x_encoded, y
 
 
def split_dataset(x_raw: pd.DataFrame, x_encoded: pd.DataFrame, y: pd.Series):
    """Faz o split treino/teste, mantendo x_train em duas versões
    (raw para o LIMA, encoded para o classificador) alinhadas pelo índice."""
    section("ETAPA 3: SEPARANDO TREINO E TESTE")
 
    idx_train, idx_test = train_test_split(
        x_raw.index,
        test_size=settings.TEST_SIZE,
        random_state=settings.RANDOM_STATE,
        stratify=y,
    )
 
    x_train_raw = x_raw.loc[idx_train]
    x_train_encoded = x_encoded.loc[idx_train]
    x_test_encoded = x_encoded.loc[idx_test]
    y_train = y.loc[idx_train]
    y_test = y.loc[idx_test]
 
    print(f"  Treino: {len(x_train_raw):,} | Teste: {len(x_test_encoded):,}")
    return x_train_raw, x_train_encoded, x_test_encoded, y_train, y_test
 
