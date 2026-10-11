from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.revenue_at_risk import add_revenue_at_risk

TIER_THRESHOLD = 0.5
ACTION_MAP = {
    'Active (<3 Mo)': 'Maintain Standard Engagement & Loyalty Rewards',
    'Moderate Risk (3-6 Mo)': 'Send Re-engagement Survey & Tailored Recommendations',
    'High Risk (6-9 Mo)': 'Trigger Urgent Direct Discount Code (High-Value Retention)',
    'Dormant (9+ Mo)': 'Route to Automated Win-Back Email Campaign',
}
_REQUIRED_KEYS = {'models', 'horizons', 'feature_columns'}


def load_model_bundle(model_path: Path) -> dict:
    """Load a bundle produced by train.save_bundle.

    joblib/pickle can execute arbitrary code: only load files you created yourself.
    """
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or not _REQUIRED_KEYS.issubset(bundle):
        raise ValueError(f'Invalid model bundle; expected keys {sorted(_REQUIRED_KEYS)}')
    return bundle


def assign_tier(p_by_horizon: list, threshold: float = TIER_THRESHOLD) -> np.ndarray:
    """Tier = longest horizon over which the customer is likely to stay inactive."""
    p90, p180, p270 = p_by_horizon
    return np.select(
        [p270 >= threshold, p180 >= threshold, p90 >= threshold],
        ['Dormant (9+ Mo)', 'High Risk (6-9 Mo)', 'Moderate Risk (3-6 Mo)'],
        default='Active (<3 Mo)')


def run_batch_inference(df_features: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    cols = bundle['feature_columns']
    missing = set(cols) - set(df_features.columns)
    if missing:
        raise ValueError(f'Missing feature columns: {sorted(missing)}')
    X = df_features[cols]
    horizons = bundle['horizons']

    probs, running = [], None
    for h in horizons:
        p = bundle['models'][h].predict_proba(X)[:, 1]
        # P(inactive for 270d) can never exceed P(inactive for 90d): enforce coherence.
        running = p if running is None else np.minimum(p, running)
        probs.append(running)

    out = df_features[['Customer_ID']].copy()
    out['Churn_Probability'] = probs[0].round(4)
    for h, p in zip(horizons[1:], probs[1:]):
        out[f'P_No_Purchase_{h}d'] = p.round(4)
    out['Is_Churned_Predicted'] = (probs[0] >= 0.5).astype(int)
    out['Risk_Horizon_Tier'] = assign_tier(probs)
    out['Recommended_Business_Action'] = out['Risk_Horizon_Tier'].map(ACTION_MAP)
    return add_revenue_at_risk(out, df_features, horizon_days=horizons[0])
