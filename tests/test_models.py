import numpy as np
import pandas as pd
import pytest

from src.features.build_rfm_features import FEATURE_COLUMNS, generate_customer_rfm_matrix
from src.models.evaluate import evaluate_model_performance
from src.models.predict import ACTION_MAP, load_model_bundle, run_batch_inference
from src.models.revenue_at_risk import add_revenue_at_risk
from src.models.train import save_bundle


def test_model_learns_signal_on_a_temporal_split(trained):
    _, metrics = trained
    for horizon, m in metrics.items():
        assert m['roc_auc'] > 0.65, f'horizon {horizon}: model barely beats chance'
        assert 0.0 <= m['brier'] <= 0.5


def test_inference_output_contract(trained, transactions):
    bundle, _ = trained
    feats = generate_customer_rfm_matrix(transactions)
    out = run_batch_inference(feats, bundle)
    assert len(out) == len(feats)
    for col in ['Customer_ID', 'Churn_Probability', 'P_No_Purchase_180d', 'P_No_Purchase_270d',
                'Is_Churned_Predicted', 'Risk_Horizon_Tier', 'Recommended_Business_Action',
                'Revenue_At_Risk_90d']:
        assert col in out.columns
    assert set(out['Risk_Horizon_Tier']) <= set(ACTION_MAP)
    assert out['Recommended_Business_Action'].notna().all()


def test_probabilities_are_coherent_across_horizons(trained, transactions):
    bundle, _ = trained
    out = run_batch_inference(generate_customer_rfm_matrix(transactions), bundle)
    assert (out['Churn_Probability'] >= out['P_No_Purchase_180d']).all()
    assert (out['P_No_Purchase_180d'] >= out['P_No_Purchase_270d']).all()


def test_missing_feature_columns_are_reported(trained, transactions):
    bundle, _ = trained
    feats = generate_customer_rfm_matrix(transactions).drop(columns=['Recency_Days'])
    with pytest.raises(ValueError, match='Recency_Days'):
        run_batch_inference(feats, bundle)


def test_bundle_roundtrip_and_validation(trained, tmp_path):
    bundle, _ = trained
    path = tmp_path / 'b.pkl'
    save_bundle(bundle, path)
    assert load_model_bundle(path)['feature_columns'] == FEATURE_COLUMNS
    import joblib
    joblib.dump({'not': 'a bundle'}, tmp_path / 'bad.pkl')
    with pytest.raises(ValueError, match='Invalid model bundle'):
        load_model_bundle(tmp_path / 'bad.pkl')


def test_revenue_at_risk_formula():
    feats = pd.DataFrame({'Monetary_Spend': [900.0, 100.0], 'Tenure_Days': [90, 5]})
    scored = pd.DataFrame({'Churn_Probability': [0.5, 1.0]})
    out = add_revenue_at_risk(scored, feats, horizon_days=90)
    # customer 1: 900/90 = 10/day -> 900 expected -> 450 at risk
    assert out['Revenue_At_Risk_90d'].iloc[0] == pytest.approx(450.0)
    # customer 2: tenure floored at 30 days -> 100/30*90 = 300
    assert out['Expected_Revenue_90d'].iloc[1] == pytest.approx(300.0)


def test_evaluate_requires_both_classes(trained, transactions):
    bundle, _ = trained
    X = generate_customer_rfm_matrix(transactions)[FEATURE_COLUMNS]
    with pytest.raises(ValueError):
        evaluate_model_performance(bundle['models'][90], X, np.zeros(len(X)))
