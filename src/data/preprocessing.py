from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from src.configs.settings import (
    PROCESSED_DATA_DIR,
    MODEL_ARTIFACTS_DIR,
    TRAIN_TEST_SPLIT,
    RANDOM_SEED,
)

def preprocess_data(
    input_filename: str,
    target_column: str,
) -> None:
    input_path = PROCESSED_DATA_DIR / input_filename

    df = pd.read_csv(input_path)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TRAIN_TEST_SPLIT,
        random_state=RANDOM_SEED,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(
        PROCESSED_DATA_DIR / "X_train.csv", index=False
    )
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(
        PROCESSED_DATA_DIR / "X_test.csv", index=False
    )
    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)

    joblib.dump(scaler, MODEL_ARTIFACTS_DIR / "scaler.pkl")


if __name__ == "__main__":
    preprocess_data(
        input_filename="dataset.csv",
        target_column="target",
    )