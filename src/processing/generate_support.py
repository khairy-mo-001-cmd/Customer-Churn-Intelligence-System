import pandas as pd

from src.utils.schema import AMOUNT, CUSTOMER, QTY, validate_schema


def generate_support_lookup_tables(df: pd.DataFrame) -> pd.DataFrame:
    """Sales summary per country (the dataset has no product-category column)."""
    validate_schema(df, {'Country', AMOUNT, QTY, CUSTOMER})
    return (df.groupby('Country')
              .agg(Total_Sales=(AMOUNT, 'sum'), Total_Units=(QTY, 'sum'),
                   Customers=(CUSTOMER, 'nunique'))
              .reset_index())
