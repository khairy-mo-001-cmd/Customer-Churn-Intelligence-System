import logging

import pandas as pd

from src.utils.schema import (AMOUNT, CUSTOMER, DATE, INVOICE, QTY, REQUIRED_CLEAN, SKU,
                              validate_schema)

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    'Recency_Days', 'Frequency_Orders', 'Monetary_Spend', 'Total_Units',
    'Unique_Products', 'Avg_Order_Value', 'Avg_Units_Per_Order', 'Tenure_Days',
]


def generate_customer_rfm_matrix(df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    """One row per customer, computed ONLY from transactions strictly before snapshot_date.

    Filtering on the snapshot guarantees that no information from the future leaks into
    the features, which is what makes forward-looking churn labels honest.
    """
    validate_schema(df, REQUIRED_CLEAN)
    if snapshot_date is None:
        snapshot = df[DATE].max() + pd.Timedelta(days=1)
    else:
        snapshot = pd.Timestamp(snapshot_date)
    past = df[df[DATE] < snapshot]
    if past.empty:
        raise ValueError(f'No transactions before snapshot date {snapshot}')

    out = past.groupby(CUSTOMER).agg(
        First_Invoice=(DATE, 'min'),
        Last_Invoice=(DATE, 'max'),
        Frequency_Orders=(INVOICE, 'nunique'),
        Monetary_Spend=(AMOUNT, 'sum'),
        Total_Units=(QTY, 'sum'),
        Unique_Products=(SKU, 'nunique'),
    )
    out['Recency_Days'] = (snapshot - out.pop('Last_Invoice')).dt.days
    out['Tenure_Days'] = (snapshot - out.pop('First_Invoice')).dt.days
    out['Avg_Order_Value'] = out['Monetary_Spend'] / out['Frequency_Orders']
    out['Avg_Units_Per_Order'] = out['Total_Units'] / out['Frequency_Orders']
    out = out.reset_index().rename(columns={CUSTOMER: 'Customer_ID'})
    logger.info('Built feature matrix for %s customers at %s', f'{len(out):,}', snapshot)
    return out[['Customer_ID'] + FEATURE_COLUMNS]
