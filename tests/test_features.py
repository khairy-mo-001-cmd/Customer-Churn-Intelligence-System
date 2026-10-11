import pandas as pd
import pytest

from src.features.build_rfm_features import FEATURE_COLUMNS, generate_customer_rfm_matrix
from src.features.build_training_set import build_labeled_set, temporal_cutoffs


@pytest.fixture
def tiny() -> pd.DataFrame:
    rows = [  # Customer ID, Invoice, date, amount, qty, sku
        (1, 'O1', '2026-01-01', 100.0, 2, 'P1'), (1, 'O2', '2026-01-15', 200.0, 3, 'P2'),
        (2, 'O3', '2026-02-01', 150.0, 1, 'P1'),
        (3, 'O4', '2026-01-05', 300.0, 4, 'P3'), (3, 'O5', '2026-02-10', 250.0, 2, 'P2'),
        (3, 'O6', '2026-03-01', 400.0, 5, 'P1'),
    ]
    return pd.DataFrame(rows, columns=['Customer ID', 'Invoice', 'InvoiceDate', 'totalprice',
                                       'Quantity', 'StockCode']).assign(
        InvoiceDate=lambda d: pd.to_datetime(d['InvoiceDate']))


def test_rfm_values_by_hand(tiny):
    out = generate_customer_rfm_matrix(tiny).set_index('Customer_ID')
    assert list(out.columns) == FEATURE_COLUMNS
    # snapshot = 2026-03-02
    assert out.loc[1, 'Recency_Days'] == 46 and out.loc[2, 'Recency_Days'] == 29
    assert out.loc[3, 'Recency_Days'] == 1
    assert out.loc[1, 'Monetary_Spend'] == 300.0 and out.loc[1, 'Frequency_Orders'] == 2
    assert out.loc[1, 'Avg_Order_Value'] == 150.0
    assert out.loc[1, 'Tenure_Days'] == 60  # first purchase Jan 1 -> snapshot Mar 2


def test_features_never_see_the_future(tiny):
    out = generate_customer_rfm_matrix(tiny, snapshot_date='2026-02-05').set_index('Customer_ID')
    assert set(out.index) == {1, 2, 3}
    assert out.loc[3, 'Frequency_Orders'] == 1  # O5 (Feb 10) and O6 are in the future
    assert out.loc[3, 'Monetary_Spend'] == 300.0


def test_labels_come_from_the_future_window(tiny):
    # cutoff Feb 5, horizon 25 days -> window [Feb 5, Mar 2): customer 3 buys, 1 and 2 don't
    labeled = build_labeled_set(tiny, '2026-02-05', 25).set_index('Customer_ID')
    assert labeled.loc[3, 'Churn_Target'] == 0
    assert labeled.loc[1, 'Churn_Target'] == 1 and labeled.loc[2, 'Churn_Target'] == 1


def test_unobservable_label_window_is_rejected(tiny):
    with pytest.raises(ValueError, match='fully observable'):
        build_labeled_set(tiny, '2026-02-05', 90)


def test_temporal_cutoffs_do_not_overlap(transactions):
    snapshot, train_cut, test_cut = temporal_cutoffs(transactions, 90)
    assert train_cut + pd.Timedelta(days=90) == test_cut
    assert test_cut + pd.Timedelta(days=90) == snapshot
