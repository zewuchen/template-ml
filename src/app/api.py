from fastapi import FastAPI
import numpy as np
import joblib

app = FastAPI(title="Simple ML API")

MODEL_PATH = "model_artifacts/model.pkl"
model = joblib.load(MODEL_PATH)


@app.get("/")
def healthcheck():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: list[float]):
    """
    Exemplo de payload:
    {
        "features": [1.0, 2.0, 3.0, 4.0]
    }
    """
    X = np.array(features).reshape(1, -1)
    prediction = model.predict(X)

    return {
        "prediction": prediction.tolist()
    }