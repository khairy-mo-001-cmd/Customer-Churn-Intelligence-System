# src/processing/generate_support.py
import pandas as pd

def generate_support_lookup_tables(df_transactions: pd.DataFrame) -> pd.DataFrame:
    """Generates supplementary lookup tables for categories or customer metadata."""
    if 'Product_Category' in df_transactions.columns:
        category_summary = df_transactions.groupby('Product_Category').agg(
            Total_Sales=('Order_Amount', 'sum'),
            Total_Quantity=('Quantity', 'sum')
        ).reset_index()
        return category_summary
    return pd.DataFrame()