from pathlib import Path
import pandas as pd

from src.configs.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR

def ingest_data(
    input_filename: str,
    output_filename: str = "dataset.csv"
) -> None:
    input_path = RAW_DATA_DIR / input_filename
    output_path = PROCESSED_DATA_DIR / output_filename

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    ingest_data("data.csv")