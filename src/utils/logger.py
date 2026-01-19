import logging
from pathlib import Path
from datetime import datetime
from src.configs.settings import BASE_DIR

# Diretório para salvar logs
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Nome do arquivo de log com timestamp
LOG_FILE = LOG_DIR / f"ml_project_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def get_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Evita adicionar múltiplos handlers se já existir
    if not logger.handlers:
        # Handler para console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Handler para arquivo
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger