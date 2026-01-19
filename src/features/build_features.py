from pathlib import Path
import pandas as pd

from src.configs.settings import PROCESSED_DATA_DIR, MODEL_ARTIFACTS_DIR


def build_features(input_filename: str, output_filename: str = "X_features.csv") -> None:
    input_path = PROCESSED_DATA_DIR / input_filename
    df = pd.read_csv(input_path)

    # Feature Engineering Simples
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        df["sum_numeric"] = df[numeric_cols].sum(axis=1)

    # Exemplo: média das colunas numéricas
    if numeric_cols:
        df["mean_numeric"] = df[numeric_cols].mean(axis=1)

    # Salva resultado
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_DIR / output_filename