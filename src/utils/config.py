"""Project paths and environment-based settings. No secrets live in this file."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
REPORTS_DIR = PROJECT_ROOT / 'reports'
PREDICTIONS_DIR = REPORTS_DIR / 'processed_predictions'

RAW_FILE = RAW_DATA_DIR / 'online_retail_II.xlsx'
CLEAN_TRANSACTIONS_PATH = PROCESSED_DATA_DIR / 'registered_transactions_clean.parquet'
CHAMPION_MODEL_PATH = MODELS_DIR / 'churn_model_bundle.pkl'
METRICS_PATH = REPORTS_DIR / 'model_metrics.json'
PREDICTIONS_PATH = PREDICTIONS_DIR / 'scored_customer_churn_predictions.parquet'


def ensure_dirs() -> None:
    """Create output folders. Called explicitly, never as an import side effect."""
    for directory in (PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, PREDICTIONS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def get_db_settings() -> dict:
    """Read DB settings from the environment. DB_PASS is mandatory (no default)."""
    password = os.getenv('DB_PASS')
    if not password:
        raise RuntimeError('DB_PASS environment variable is required (see .env.example)')
    return {
        'user': os.getenv('DB_USER', 'postgres'),
        'password': password,
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'database': os.getenv('DB_NAME', 'customer_churn'),
    }
