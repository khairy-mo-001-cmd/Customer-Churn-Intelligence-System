import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_raw_transactions(file_path: Path) -> pd.DataFrame:
    """Load raw transactions from xlsx (all sheets are concatenated), csv or parquet."""
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f'Raw transaction file not found at: {file_path}')

    suffix = file_path.suffix.lower()
    if suffix in ('.xlsx', '.xls'):
        sheets = pd.read_excel(file_path, sheet_name=None)
        df = pd.concat(sheets.values(), ignore_index=True)
    elif suffix == '.csv':
        df = pd.read_csv(file_path)
    elif suffix in ('.parquet', '.pq'):
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f'Unsupported file format: {file_path.suffix}')

    logger.info('Loaded %s raw transaction records', f'{len(df):,}')
    return df
