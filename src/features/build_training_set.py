import pandas as pd

from src.features.build_rfm_features import generate_customer_rfm_matrix
from src.utils.schema import CUSTOMER, DATE


def build_labeled_set(df: pd.DataFrame, cutoff, horizon_days: int) -> pd.DataFrame:
    """Features from the past (< cutoff), target from the future window [cutoff, cutoff+H).

    Churn_Target = 1 when the customer made NO purchase inside the future window.
    """
    cutoff = pd.Timestamp(cutoff)
    end = cutoff + pd.Timedelta(days=horizon_days)
    if end > df[DATE].max() + pd.Timedelta(days=1):
        raise ValueError('The future window extends past the available data: '
                         'labels would not be fully observable')
    features = generate_customer_rfm_matrix(df, snapshot_date=cutoff)
    future = df[(df[DATE] >= cutoff) & (df[DATE] < end)]
    returned = set(future[CUSTOMER].unique())
    features['Churn_Target'] = (~features['Customer_ID'].isin(returned)).astype('int8')
    return features


def temporal_cutoffs(df: pd.DataFrame, horizon_days: int):
    """Return (snapshot, train_cutoff, test_cutoff).

    test_cutoff = snapshot - H  -> its label window ends exactly at the last data point.
    train_cutoff = test_cutoff - H -> train label window ends where test features end,
    so training never sees outcomes that overlap the test period.
    """
    snapshot = df[DATE].max() + pd.Timedelta(days=1)
    test_cutoff = snapshot - pd.Timedelta(days=horizon_days)
    train_cutoff = test_cutoff - pd.Timedelta(days=horizon_days)
    return snapshot, train_cutoff, test_cutoff
