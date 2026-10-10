# src/ingestion/download_raw.py
import pandas as pd
from pathlib import Path
from src.utils.config import DATA_DIR

def load_raw_transactions(file_path: Path) -> pd.DataFrame:
    """Loads raw transaction logs from CSV or Parquet format with validation."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Raw transaction file not found at: {file_path}")
    
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix in ['.parquet', '.pq']:
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
    print(f"[OK] Successfully loaded {len(df):,} raw transaction records.")
    return df