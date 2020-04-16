from sqlalchemy.engine import create_engine
from sqlalchemy.sql import text

from elt_tools.settings import DATABASES

OLTP_ENGINE = 'oltp_engine'
BIGQUERY_ENGINE = 'bigquery_engine'
REDSHIFT_ENGINE = 'redshift_engine'


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
}


def engine_from_settings(db_key, databases=None):
    # Use settings that are passed in,
    # otherwise read from settings module
    if not databases:
        databases = DATABASES
    config = databases[db_key]
    engine_func = engine_key_to_engine[config['engine']]
    del config['engine']
    engine = engine_func(**config)
    return engine

