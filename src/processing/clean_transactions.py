import logging

import pandas as pd

from src.utils.schema import AMOUNT, CUSTOMER, DATE, INVOICE, PRICE, QTY, SKU, validate_schema

logger = logging.getLogger(__name__)

ADMIN_CODES = {'POST', 'D', 'M', 'BANK CHARGES', 'AMAZONFEE', 'MANUAL', 'TEST', 'CRUK'}


def clean_transaction_logs(df: pd.DataFrame, registered_only: bool = True) -> pd.DataFrame:
    """Clean raw Online Retail II lines.

    Removes exact duplicates, rows without invoice/date, cancellations (invoice starts
    with 'C'), non-positive quantity/price and administrative SKUs. Missing values are
    never imputed with fake numbers. With registered_only=True, guest checkouts
    (no Customer ID) are dropped because churn is defined per customer.
    """
    validate_schema(df)
    initial = len(df)
    df = df.drop_duplicates().copy()
    # Excel yields mixed int/str in these columns, which parquet cannot store.
    df[INVOICE] = df[INVOICE].astype(str).str.strip()
    df[SKU] = df[SKU].astype(str).str.strip()

    df[DATE] = pd.to_datetime(df[DATE], errors='coerce')
    df = df.dropna(subset=[INVOICE, DATE])
    df[QTY] = pd.to_numeric(df[QTY], errors='coerce')
    df[PRICE] = pd.to_numeric(df[PRICE], errors='coerce')
    df = df.dropna(subset=[QTY, PRICE])

    sku = df[SKU].str.upper()
    cancelled = df[INVOICE].str.upper().str.startswith('C')
    valid = ~cancelled & ~sku.isin(ADMIN_CODES) & (df[QTY] > 0) & (df[PRICE] > 0)
    df = df[valid].copy()

    if registered_only:
        df = df.dropna(subset=[CUSTOMER])
        df[CUSTOMER] = df[CUSTOMER].astype('int64')

    df[QTY] = df[QTY].astype('int64')
    df[AMOUNT] = df[QTY] * df[PRICE]
    logger.info('Cleaning removed %s of %s rows', f'{initial - len(df):,}', f'{initial:,}')
    return df.reset_index(drop=True)
