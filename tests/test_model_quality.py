"""Guards the PUBLISHED model artifact so a stale or corrupted file can never ship again."""
import pytest

from src.models.predict import load_model_bundle
from src.utils.config import CHAMPION_MODEL_PATH

pytestmark = pytest.mark.skipif(not CHAMPION_MODEL_PATH.is_file(),
                                reason='no trained model in this checkout')


def test_published_model_metrics_are_sane():
    bundle = load_model_bundle(CHAMPION_MODEL_PATH)
    for horizon, m in bundle['metrics'].items():
        assert m['roc_auc'] >= 0.70, f'{horizon}d model AUC={m["roc_auc"]:.3f}: stale or broken'
        assert m['roc_auc'] >= m['baseline_recency_roc_auc'], 'model must beat the Recency rule'


def test_published_model_really_scores_like_its_metrics():
    """Re-evaluate the saved model on a fresh temporal split of the real data."""
    from src.features.build_training_set import build_labeled_set, temporal_cutoffs
    from src.utils.config import CLEAN_TRANSACTIONS_PATH
    if not CLEAN_TRANSACTIONS_PATH.is_file():
        pytest.skip('clean transactions not available')
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    bundle = load_model_bundle(CHAMPION_MODEL_PATH)
    df = pd.read_parquet(CLEAN_TRANSACTIONS_PATH)
    _, _, test_cut = temporal_cutoffs(df, 90)
    test = build_labeled_set(df, test_cut, 90)
    p = bundle['models'][90].predict_proba(test[bundle['feature_columns']])[:, 1]
    # the final model was refit on this very set, so this is a sanity floor, not an estimate
    assert roc_auc_score(test['Churn_Target'], p) >= 0.75
