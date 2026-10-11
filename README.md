# Customer Churn Intelligence System

Predicts **implicit customer churn** for a non-subscription e-commerce business and ranks
customers by **revenue at risk**, using forward-looking labels and strict temporal validation.

Data: the *Online Retail II* transaction log (UK online retailer, Dec 2009 - Dec 2011).
There is no cancellation event in this kind of business, so "churn" is defined by behaviour:

> **Churn_Target = 1** when a customer makes **no purchase in the next H days** (H = 90, 180, 270).

## Results (temporal hold-out, honest numbers)

Each model is trained on customers as they looked at cutoff `snapshot - 2H`, and evaluated on
customers at cutoff `snapshot - H`, so the test labels come from a later period than anything
the model saw. A "Recency only" rule (score = days since last purchase) is the baseline to beat.

| Horizon | Test customers | Churn rate | ROC-AUC | Recency-only AUC | PR-AUC | Brier |
|---|---|---|---|---|---|---|
| 90 days  | 5,270 | 56.7% | **0.794** | 0.763 | 0.813 | 0.198 |
| 180 days | 4,967 | 48.2% | **0.789** | 0.771 | 0.746 | 0.196 |
| 270 days | 4,601 | 41.8% | **0.764** | 0.742 | 0.654 | 0.226 |

**How to read this:** the model ranks a churner above a non-churner about 76-79% of the time.
Gradient boosting adds a *modest* but consistent lift (+1.7 to +3 AUC points) over the simple
Recency rule, which is already strong. Most of the signal in this dataset is recency itself.
Full metrics: `reports/model_metrics.json` (produced by `python -m src.pipeline train`).

Latest scoring run (snapshot 2011-12-10, 5,862 customers) assigns tiers from the three horizons:
Active, Moderate Risk (likely inactive >= 3 months), High Risk (>= 6 months) and
Dormant (>= 9 months), each mapped to a recommended retention action.
`Revenue_At_Risk_90d = P(no purchase in 90d) x expected 90-day revenue`, where expected revenue
projects the customer's historical average daily spend (a deliberately simple estimate).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                                      # only needed for PostgreSQL

# put online_retail_II.xlsx in data/raw/, then:
python -m src.pipeline clean      # raw xlsx -> data/processed/registered_transactions_clean.parquet
python -m src.pipeline train      # temporal validation + model bundle + reports/model_metrics.json
python -m src.pipeline score      # reports/processed_predictions/scored_customer_churn_predictions.parquet
python -m src.pipeline score --to-postgres                # optional, needs DB_* variables

python -m pytest -q
```

### API

```bash
uvicorn src.api.app:app --reload        # interactive docs at http://localhost:8000/docs
```

`POST /predict` takes the 8 customer features (`Recency_Days`, `Frequency_Orders`,
`Monetary_Spend`, `Total_Units`, `Unique_Products`, `Avg_Order_Value`, `Avg_Units_Per_Order`,
`Tenure_Days`) and returns churn probabilities for 90/180/270 days, the risk tier, the
recommended action and the revenue at risk. Invalid input returns `422`; a missing model returns `503`.

### Docker

```bash
python -m src.pipeline train && docker build -t churn-api . && docker run -p 8000:8000 churn-api
```

## Project layout

```
src/
  utils/        config (env-based, no secrets), schema (column names), db connector
  ingestion/    xlsx/csv/parquet loader, PostgreSQL writer
  processing/   transaction cleaning, country summary
  features/     customer features (past only), forward-looking labelled sets
  models/       training + temporal validation, evaluation, inference, revenue at risk
  api/          FastAPI service
  pipeline.py   CLI: clean | train | score
tests/          unit, integration and model-quality tests (synthetic data in the real schema)
notebooks/      exploratory notebooks (the first four pipeline notebooks are legacy, see banner)
```

## Design decisions

- **Features never see the future.** `generate_customer_rfm_matrix` filters transactions strictly
  before the snapshot date, and labels come only from the following window.
- **No scaler**: tree models are invariant to feature scaling.
- **No invented data**: missing prices/quantities are dropped, never filled with 0.
- **Probabilities are calibrated** (isotonic) when that lowers the Brier score on the test period.
- **Coherent horizons**: P(inactive 270d) is forced to be <= P(inactive 90d).
- **Atomic model writes** and a **model-quality test** that fails if a stale artifact is published.

## Limitations

- One retailer, two years of data; the 270-day model is trained on fewer customers.
- Revenue at risk uses average historical spend; BG/NBD + Gamma-Gamma would be more rigorous.
- The tier thresholds (0.5) are not tuned to retention costs; choose them with business input.
- Model files are pickles: only load files you trained yourself.
- Not validated against a live PostgreSQL server or a Docker build in CI.
