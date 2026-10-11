from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine

from src.utils.config import get_db_settings


def get_postgres_engine() -> Engine:
    """Build a SQLAlchemy engine. URL.create escapes special characters in the password."""
    s = get_db_settings()
    url = URL.create(
        'postgresql+psycopg',
        username=s['user'], password=s['password'],
        host=s['host'], port=s['port'], database=s['database'],
    )
    return create_engine(url, pool_pre_ping=True, connect_args={'connect_timeout': 10})
