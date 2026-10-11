import pytest
from fastapi.testclient import TestClient

from src.api import app as api_module
from src.models.train import save_bundle

PAYLOAD = {'Customer_ID': 'C1', 'Recency_Days': 40, 'Frequency_Orders': 5,
           'Monetary_Spend': 500.0, 'Total_Units': 10, 'Unique_Products': 3,
           'Avg_Order_Value': 100.0, 'Avg_Units_Per_Order': 2.0, 'Tenure_Days': 200}


@pytest.fixture
def client_with_model(trained, tmp_path, monkeypatch):
    path = tmp_path / 'model.pkl'
    save_bundle(trained[0], path)
    monkeypatch.setattr(api_module, 'MODEL_PATH', path)
    with TestClient(api_module.app) as client:  # `with` runs the lifespan (loads the model)
        yield client


@pytest.fixture
def client_without_model(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, 'MODEL_PATH', tmp_path / 'missing.pkl')
    with TestClient(api_module.app) as client:
        yield client


def test_health_with_model(client_with_model):
    body = client_with_model.get('/health').json()
    assert body['model_loaded'] is True and body['trained_at']


def test_health_without_model(client_without_model):
    assert client_without_model.get('/health').json()['model_loaded'] is False


def test_predict_success_path(client_with_model):
    r = client_with_model.post('/predict', json=PAYLOAD)
    assert r.status_code == 200
    pred = r.json()['prediction']
    assert 0.0 <= pred['Churn_Probability'] <= 1.0
    assert pred['Risk_Horizon_Tier'] in {'Active (<3 Mo)', 'Moderate Risk (3-6 Mo)',
                                         'High Risk (6-9 Mo)', 'Dormant (9+ Mo)'}
    assert pred['Revenue_At_Risk_90d'] >= 0


def test_predict_without_model_is_503(client_without_model):
    assert client_without_model.post('/predict', json=PAYLOAD).status_code == 503


def test_invalid_payloads_are_rejected(client_with_model):
    assert client_with_model.post('/predict', json={'Customer_ID': 'C1'}).status_code == 422
    assert client_with_model.post('/predict', json={**PAYLOAD, 'Frequency_Orders': -5}).status_code == 422
    assert client_with_model.post('/predict', json={**PAYLOAD, 'Tenure_Days': 10}).status_code == 422
