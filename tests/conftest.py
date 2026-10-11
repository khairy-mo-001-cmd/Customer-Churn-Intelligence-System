"""Shared fixtures. Synthetic data uses the REAL schema of the Online Retail II data."""
import numpy as np
import pandas as pd
import pytest

from src.models.train import train_horizon_models


def make_transactions(n_customers: int = 400, days: int = 700, seed: int = 7) -> pd.DataFrame:
    """Customers buy at a steady rate until a random 'churn day', then stop."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp('2010-01-01')
    rows, invoice = [], 100000
    for cid in range(10000, 10000 + n_customers):
        first = int(rng.integers(0, days // 2))
        lifetime = int(rng.exponential(220)) + 30
        rate = rng.uniform(0.02, 0.15)  # purchases per day
        for day in range(first, min(first + lifetime, days)):
            if rng.random() < rate:
                invoice += 1
                for _ in range(int(rng.integers(1, 4))):
                    qty = int(rng.integers(1, 10))
                    price = round(float(rng.uniform(1, 20)), 2)
                    rows.append({
                        'Invoice': str(invoice), 'StockCode': f'S{int(rng.integers(1, 60))}',
                        'Description': 'ITEM', 'Quantity': qty,
                        'InvoiceDate': start + pd.Timedelta(days=day, hours=10),
                        'Price': price, 'Customer ID': cid, 'Country': 'United Kingdom',
                        'totalprice': qty * price,
                    })
    return pd.DataFrame(rows)


@pytest.fixture(scope='session')
def transactions() -> pd.DataFrame:
    return make_transactions()


@pytest.fixture(scope='session')
def trained(transactions):
    bundle, metrics = train_horizon_models(transactions)
    return bundle, metrics


@pytest.fixture
def raw_dirty() -> pd.DataFrame:
    """A tiny raw frame with every kind of dirt the cleaner must handle."""
    ts = '2011-01-05 10:00:00'
    cols = ['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'Price',
            'Customer ID', 'Country']
    rows = [
        ['A1', '85048', 'OK line', 2, ts, 5.0, 12345.0, 'UK'],
        ['A1', '85048', 'OK line', 2, ts, 5.0, 12345.0, 'UK'],      # exact duplicate
        ['A2', '22423', 'second', 1, ts, 10.0, 12345.0, 'UK'],
        ['C3', '22423', 'cancelled', -1, ts, 10.0, 12345.0, 'UK'],  # cancellation
        ['A4', 'POST', 'postage', 1, ts, 18.0, 12345.0, 'UK'],      # admin SKU
        ['A5', '22111', 'free item', 1, ts, 0.0, 12345.0, 'UK'],    # zero price
        ['A6', '22222', 'guest', 1, ts, 3.0, None, 'UK'],           # guest checkout
        ['A7', '22333', 'bad date', 1, 'not-a-date', 3.0, 12345.0, 'UK'],
    ]
    return pd.DataFrame(rows, columns=cols)
