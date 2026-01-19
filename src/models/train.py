from pathlib import Path
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

from src.configs.settings import PROCESSED_DATA_DIR, MODEL_ARTIFACTS_DIR, MODEL_PARAMS

def train_model(
    X_train_filename: str = "X_train.csv",
    y_train_filename: str = "y_train.csv",
    model_filename: str = "model.pkl"
) -> None:
    # Carrega dados
    X_train = pd.read_csv(PROCESSED_DATA_DIR / X_train_filename)
    y_train = pd.read_csv(PROCESSED_DATA_DIR / y_train_filename).squeeze()

    # Cria modelo
    model = LogisticRegression(**MODEL_PARAMS)

    # Treina modelo
    model.fit(X_train, y_train)

    # Cria diretório caso não exista
    MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Salva modelo
    model_path = MODEL_ARTIFACTS