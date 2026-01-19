from src.data.ingest import ingest_data
from src.data.preprocessing import preprocess_data
from src.features.build_features import build_features
from src.models.train import train_model
from src.evaluation.evaluate import evaluate_model

def run_pipeline(
    raw_data_filename: str = "data.csv",
    target_column: str = "target"
):
    """
    Pipeline completo de treinamento:
    1. Ingestão de dados
    2. Pré-processamento
    3. Engenharia de features
    4. Treinamento de modelo
    5. Avaliação
    """
    print("==== STEP 1: Ingest Data ====")
    ingest_data(input_filename=raw_data_filename)

    print("==== STEP 2: Preprocess Data ====")
    preprocess_data(input_filename="dataset.csv", target_column=target_column)

    print("==== STEP 3: Build Features ====")
    build_features(input_filename="X_train.csv", output_filename="X_train_features.csv")
    build_features(input_filename="X_test.csv", output_filename="X_test_features.csv")