from unittest.mock import  MagicMock
from sqlalchemy.engine import create_engine
from sqlalchemy.sql import text

from elt_tools.settings_example import DATABASES

OLTP_ENGINE = 'oltp_engine'
BIGQUERY_ENGINE = 'bigquery_engine'
REDSHIFT_ENGINE = 'redshift_engine'
MOCK_ENGINE = 'mock_engine'



def mock_engine():
    return MagicMock()

def bigquery_engine(gcp_project=None, dataset_id=None, gcp_credentials=None):
    bigquery_uri = f'bigquery://{gcp_project}/{dataset_id}'
    engine = create_engine(bigquery_uri, credentials_path=gcp_credentials)
    return engine


def redshift_engine(sql_alchemy_conn_string=None, default_schema='public', connect_timeout=3600):
    engine = create_engine(sql_alchemy_conn_string, connect_args={'connect_timeout': connect_timeout})
    engine.execute(text('SET search_path TO %s,public;' % default_schema).execution_options(
        autocommit=True))
    return engine


def oltp_engine(sql_alchemy_conn_string=None, connect_timeout=3600):
    engine = create_engine(sql_alchemy_conn_string, connect_args={'connect_timeout': connect_timeout})
    return engine


engine_key_to_engine = {
    OLTP_ENGINE: oltp_engine,
    BIGQUERY_ENGINE: bigquery_engine,
    REDSHIFT_ENGINE: redshift_engine,
    MOCK_ENGINE: mock_engine,
}


def engine_from_settings(db_key, database_settings=None):
    # Use settings that are passed in,
    # otherwise read from settings module
    if not database_settings:
        database_settings = DATABASES
    config = database_settings[db_key]
    engine_func = engine_key_to_engine[config['engine']]
    del config['engine']
    engine = engine_func(**config)
    return engine

