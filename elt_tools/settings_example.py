from os import environ

DATABASES = {
    'db_key11': {
        'engine': 'oltp_engine',
        'sql_alchemy_conn_string': environ.get('mysql_db_uri'),
    },
    'db_key12': {
        'engine': 'redshift_engine',
        'sql_alchemy_conn_string': environ.get('redshift_db_uri'),
        'default_schema': 'myschema',
    },
    'db_key21': {
        'engine': 'oltp_engine',
        'sql_alchemy_conn_string': environ.get('postgres_db_uri'),
    },
    'db_key22': {
        'engine': 'bigquery_engine',
        'dataset_id': 'mydata',
        'gcp_project': environ.get('GCP_PROJECT'),
        'gcp_credentials': environ.get('GOOGLE_APPLICATION_CREDENTIALS'),
    },
}

ELT_PAIRS = {
    'pair1': {
        'source': 'db_key11', 'target': 'db_key12'
    },
    'pair2': {
        'source': 'db_key21', 'target': 'db_key22'
    },
}
