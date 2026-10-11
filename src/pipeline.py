"""Command line entry point:  python -m src.pipeline {clean|train|score}"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.features.build_rfm_features import generate_customer_rfm_matrix
from src.ingestion.download_raw import load_raw_transactions
from src.models.predict import load_model_bundle, run_batch_inference
from src.models.train import save_bundle, train_horizon_models
from src.processing.clean_transactions import clean_transaction_logs
from src.utils import config
from src.utils.schema import DATE, REQUIRED_CLEAN, validate_schema

logger = logging.getLogger('pipeline')


def cmd_clean(args) -> None:
    config.ensure_dirs()
    raw = load_raw_transactions(Path(args.raw))
    clean = clean_transaction_logs(raw, registered_only=True)
    clean.to_parquet(config.CLEAN_TRANSACTIONS_PATH, index=False)
    logger.info('Saved %s rows to %s', f'{len(clean):,}', config.CLEAN_TRANSACTIONS_PATH)


def _load_clean() -> pd.DataFrame:
    df = pd.read_parquet(config.CLEAN_TRANSACTIONS_PATH)
    validate_schema(df, REQUIRED_CLEAN)
    df[DATE] = pd.to_datetime(df[DATE])
    return df


def cmd_train(args) -> None:
    config.ensure_dirs()
    bundle, metrics = train_horizon_models(_load_clean())
    save_bundle(bundle, config.CHAMPION_MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=str))
    logger.info('Model saved to %s', config.CHAMPION_MODEL_PATH)


def cmd_score(args) -> None:
    config.ensure_dirs()
    df = _load_clean()
    bundle = load_model_bundle(config.CHAMPION_MODEL_PATH)
    features = generate_customer_rfm_matrix(df)  # snapshot = last date + 1 day
    scored = run_batch_inference(features, bundle)
    scored.to_parquet(config.PREDICTIONS_PATH, index=False)
    logger.info('Scored %s customers -> %s', f'{len(scored):,}', config.PREDICTIONS_PATH)
    if args.to_postgres:
        from src.ingestion.load_to_postgres import sync_dataframe_to_postgres
        sync_dataframe_to_postgres(scored, 'customer_churn_predictions', 'replace')


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description='Customer Churn Intelligence pipeline')
    sub = parser.add_subparsers(dest='command', required=True)
    p_clean = sub.add_parser('clean', help='raw xlsx -> clean parquet')
    p_clean.add_argument('--raw', default=str(config.RAW_FILE))
    p_clean.set_defaults(func=cmd_clean)
    sub.add_parser('train', help='train + temporally validate + save model').set_defaults(func=cmd_train)
    p_score = sub.add_parser('score', help='score all customers at the latest snapshot')
    p_score.add_argument('--to-postgres', action='store_true')
    p_score.set_defaults(func=cmd_score)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
