from pathlib import Path
import os
import random

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_ARTIFACTS_DIR = BASE_DIR / "model_artifacts"
MODEL_PATH = MODEL_ARTIFACTS_DIR / "model.pkl"

# Environment
ENV = os.getenv("ENV", "development")
DEBUG = ENV == "development"

# Random Seed
RANDOM_SEED = 42

# Training Parameters
TRAIN_TEST_SPLIT = 0.2

MODEL_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_SEED
}