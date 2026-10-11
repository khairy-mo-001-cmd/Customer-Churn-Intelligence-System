import pytest
from sqlalchemy.engine import Engine

from src.utils import config
from src.utils.db_connector import get_postgres_engine
from src.utils.schema import validate_schema


def test_config_has_no_import_side_effects_and_ensure_dirs_works(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'MODELS_DIR', tmp_path / 'm')
    monkeypatch.setattr(config, 'PROCESSED_DATA_DIR', tmp_path / 'p')
    monkeypatch.setattr(config, 'REPORTS_DIR', tmp_path / 'r')
    monkeypatch.setattr(config, 'PREDICTIONS_DIR', tmp_path / 'r' / 'pred')
    config.ensure_dirs()
    assert (tmp_path / 'm').is_dir() and (tmp_path / 'r' / 'pred').is_dir()


def test_db_password_is_mandatory(monkeypatch):
    monkeypatch.delenv('DB_PASS', raising=False)
    with pytest.raises(RuntimeError, match='DB_PASS'):
        config.get_db_settings()


def test_engine_is_built_and_special_chars_in_password_survive(monkeypatch):
    monkeypatch.setenv('DB_PASS', 'p@ss/w#rd')
    engine = get_postgres_engine()
    assert isinstance(engine, Engine)
    assert engine.url.password == 'p@ss/w#rd'
    assert engine.url.drivername == 'postgresql+psycopg'
    engine.dispose()


def test_validate_schema_reports_missing_columns(raw_dirty):
    validate_schema(raw_dirty)
    with pytest.raises(ValueError, match='Price'):
        validate_schema(raw_dirty.drop(columns=['Price']))
