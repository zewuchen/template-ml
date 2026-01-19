from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from src.configs.settings import PROCESSED_DATA_DIR, MODEL_ARTIFACTS_DIR

def evaluate_model(
    model_filename: str = "model.pkl",
    problem_type: str = "classification"  # "classification" ou "regression"
) -> None:
    # Carrega modelo
    model_path = MODEL_ARTIFACTS_DIR / model_filename
    model = joblib.load(model_path)

    # Carrega dados de teste
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").squeeze()  # transforma em Series

    # Predição
    y_pred = model.predict(X_test)

    # Avaliação
    if problem_type == "classification":
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
    elif problem_type == "regression":
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")
    else:
        raise ValueError("problem_type must be 'classification' or 'regression'")


if __name__ == "__main__":
    evaluate_model(model_filename="model.pkl", problem_type="classification")