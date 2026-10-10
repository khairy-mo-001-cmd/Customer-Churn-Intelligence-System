# src/utils/db_connector.py
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from src.utils.config import DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME

def get_postgres_engine() -> Engine:
    """Constructs and returns a SQLAlchemy production engine for PostgreSQL."""
    database_url = f"postgresql+psycopg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(
        database_url,
        pool_pre_ping=True,
        connect_args={'connect_timeout': 10}
    )
    return engine