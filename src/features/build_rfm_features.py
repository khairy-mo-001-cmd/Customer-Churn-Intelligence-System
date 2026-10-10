# src/features/build_rfm_features.py
import pandas as pd

def generate_customer_rfm_matrix(df_transactions: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw transaction history into a vectorized RFM customer feature matrix."""
    snapshot_date = df_transactions['Order_Date'].max() + pd.Timedelta(days=1)
    
    rfm_matrix = df_transactions.groupby('Customer_ID').agg(
        Recency_Days=('Order_Date', lambda x: (snapshot_date - x.max()).days),
        Frequency_Orders=('Order_ID', 'nunique'),
        Monetary_Spend=('Order_Amount', 'sum'),
        Total_Units=('Quantity', 'sum'),
        Unique_Products=('Product_ID', 'nunique')
    ).reset_index()
    
    # Advanced feature ratios
    rfm_matrix['Avg_Order_Value'] = rfm_matrix['Monetary_Spend'] / rfm_matrix['Frequency_Orders'].replace(0, 1)
    rfm_matrix['Avg_Units_Per_Order'] = rfm_matrix['Total_Units'] / rfm_matrix['Frequency_Orders'].replace(0, 1)
    
    print(f"[OK] Generated RFM feature matrix for {len(rfm_matrix):,} unique customers.")
    return rfm_matrix