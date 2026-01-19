from pathlib import Path
import pandas as pd
import joblib

from src.configs.settings import PROCESSED_DATA_DIR, MODEL_ARTIFACTS_DIR

def predict(input_filename: str,
            model_filename: str = "model.pkl",
            output_filename: str = "predictions.csv") -> None:
    # Carrega modelo
    model_path = MODEL_ARTIFACTS_DIR / model_filename
    model = joblib.load(model_path)

    # Carrega features
    input_path = PROCESSED_DATA_DIR / input_filename
    X = pd.read_csv(input_path)

    # Predição
    predictions = model.predict(X)

    # Salva previsões
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / output_filename
    pd.DataFrame(predictions, columns=["prediction"]).to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    predict(input_filename="X_test.csv", model_filename="model.pkl", output_filename="predictions.csv")