# src/processing/clean_transactions.py
import pandas as pd

def clean_transaction_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans raw transactions by handling missing values, duplicates, and type casting."""
    initial_count = len(df)
    
    # Drop duplicates
    df = df.drop_duplicates().copy()
    
    # Drop rows with critical missing identifiers
    critical_cols = ['Customer_ID', 'Order_ID', 'Order_Date']
    df = df.dropna(subset=[col for col in critical_cols if col in df.columns])
    
    # Standardize types
    if 'Order_Date' in df.columns:
        df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
    
    if 'Order_Amount' in df.columns:
        df['Order_Amount'] = pd.to_numeric(df['Order_Amount'], errors='coerce').fillna(0.0)
        
    if 'Quantity' in df.columns:
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(1).astype(int)
        
    print(f"[OK] Cleaned transactions: {initial_count - len(df):,} invalid rows removed.")
    return df