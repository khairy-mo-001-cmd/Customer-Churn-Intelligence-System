import logging

import pandas as pd

from src.utils.db_connector import get_postgres_engine

logger = logging.getLogger(__name__)
VALID_ACTIONS = {'fail', 'replace', 'append'}


def sync_dataframe_to_postgres(df: pd.DataFrame, table_name: str,
                               if_exists_action: str = 'append') -> None:
    """Write a DataFrame to PostgreSQL inside one transaction (all-or-nothing)."""
    if if_exists_action not in VALID_ACTIONS:
        raise ValueError(f'if_exists_action must be one of {sorted(VALID_ACTIONS)}')
    engine = get_postgres_engine()
    try:
        with engine.begin() as connection:
            df.to_sql(name=table_name, con=connection, if_exists=if_exists_action,
                      index=False, chunksize=5_000)
        logger.info("Synced %s rows to table '%s'", f'{len(df):,}', table_name)
    finally:
        engine.dispose()
