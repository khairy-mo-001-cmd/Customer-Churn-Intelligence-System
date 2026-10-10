# src/ingestion/load_to_postgres.py
import pandas as pd
from sqlalchemy import text
from src.utils.db_connector import get_postgres_engine

def sync_dataframe_to_postgres(df: pd.DataFrame, table_name: str, if_exists_action: str = 'replace') -> None:
    """Persists a pandas DataFrame into PostgreSQL using chunked atomic execution."""
    engine = get_postgres_engine()
    try:
        with engine.begin() as connection:
            df.to_sql(
                name=table_name,
                con=connection,
                if_exists=if_exists_action,
                index=False,
                chunksize=5_000
            )
        print(f"[OK] Successfully synced {len(df):,} rows to table '{table_name}'.")
    except Exception as e:
        print(f"[ERROR] Failed to sync DataFrame to PostgreSQL: {e}")
        raise
    finally:
        engine.dispose()