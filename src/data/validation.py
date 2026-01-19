from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

from src.configs.settings import PROCESSED_DATA_DIR, MODEL_ARTIFACTS_DIR


def validate_model(model_filename: str = "model.pkl") -> None:
    # Carrega modelo
    model_path = MODEL_ARTIFACTS_DIR / model_filename
    model = joblib.load(model_path)

    # Carrega dados de teste
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").squeeze()

    # Faz predição
    y_pred = model.predict(X_test)

    # Avalia
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    validate_model()