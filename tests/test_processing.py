import pytest

from src.processing.clean_transactions import clean_transaction_logs
from src.processing.generate_support import generate_support_lookup_tables


def test_cleaning_removes_every_kind_of_dirt(raw_dirty):
    out = clean_transaction_logs(raw_dirty)
    # only 'A1' (one copy) and 'A2' survive
    assert sorted(out['Invoice'].tolist()) == ['A1', 'A2']
    assert out['Customer ID'].dtype == 'int64'
    assert out['totalprice'].tolist() == [10.0, 10.0]


def test_cleaning_keeps_guests_when_asked(raw_dirty):
    out = clean_transaction_logs(raw_dirty, registered_only=False)
    assert 'A6' in set(out['Invoice'])


def test_missing_amounts_are_not_invented(raw_dirty):
    raw_dirty.loc[2, 'Price'] = None  # unknown price must NOT become 0 revenue
    out = clean_transaction_logs(raw_dirty)
    assert 'A2' not in set(out['Invoice'])


def test_cleaning_validates_schema(raw_dirty):
    with pytest.raises(ValueError):
        clean_transaction_logs(raw_dirty.drop(columns=['Invoice']))


def test_country_summary(transactions):
    out = generate_support_lookup_tables(transactions)
    assert list(out['Country']) == ['United Kingdom']
    assert out['Total_Sales'].iloc[0] == pytest.approx(transactions['totalprice'].sum())


def test_mixed_type_excel_columns_can_be_saved_to_parquet(tmp_path):
    """Regression: Excel gives int invoices/SKUs next to str ones; parquet needs one type."""
    import pandas as pd
    raw = pd.DataFrame({
        'Invoice': [489434, 'C489435', 489436], 'StockCode': [85048, '79323P', 22423],
        'Description': ['a', 'b', 'c'], 'Quantity': [1, -1, 2],
        'InvoiceDate': pd.to_datetime(['2011-01-05'] * 3), 'Price': [2.0, 2.0, 3.0],
        'Customer ID': [1.0, 1.0, 2.0], 'Country': ['UK'] * 3})
    out = clean_transaction_logs(raw)
    out.to_parquet(tmp_path / 'x.parquet')  # must not raise
    assert out['StockCode'].map(type).eq(str).all()
    assert sorted(out['Invoice']) == ['489434', '489436']
