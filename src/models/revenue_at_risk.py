import pandas as pd


def add_revenue_at_risk(scored: pd.DataFrame, features: pd.DataFrame,
                        horizon_days: int = 90, min_tenure_days: int = 30) -> pd.DataFrame:
    """Revenue_At_Risk = P(no purchase in horizon) x expected revenue in that horizon.

    Expected revenue is the customer's historical average daily spend
    (Monetary_Spend / Tenure_Days) projected over the horizon. This is a deliberately
    simple estimate; BG/NBD + Gamma-Gamma would be the next upgrade.
    """
    if len(scored) != len(features):
        raise ValueError('scored and features must have the same number of rows')
    out = scored.copy()
    tenure = features['Tenure_Days'].clip(lower=min_tenure_days).to_numpy()
    daily_value = features['Monetary_Spend'].to_numpy() / tenure
    out[f'Expected_Revenue_{horizon_days}d'] = (daily_value * horizon_days).round(2)
    out[f'Revenue_At_Risk_{horizon_days}d'] = (
        out['Churn_Probability'].to_numpy() * daily_value * horizon_days).round(2)
    return out
