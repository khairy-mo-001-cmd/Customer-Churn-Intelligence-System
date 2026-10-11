"""Single source of truth for column names of the Online Retail II data."""
import pandas as pd

CUSTOMER = 'Customer ID'
INVOICE = 'Invoice'
DATE = 'InvoiceDate'
AMOUNT = 'totalprice'
QTY = 'Quantity'
PRICE = 'Price'
SKU = 'StockCode'

REQUIRED_RAW = {CUSTOMER, INVOICE, DATE, PRICE, QTY, SKU}
REQUIRED_CLEAN = {CUSTOMER, INVOICE, DATE, AMOUNT, QTY, SKU}


def validate_schema(df: pd.DataFrame, required=REQUIRED_RAW) -> None:
    """Raise a clear error if expected columns are missing."""
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
