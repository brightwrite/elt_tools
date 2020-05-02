import datetime
from elt_tools.client import _construct_where_clause_from_timerange, DataClientFactory, DataClient, ELTDBPairFactory, \
    ELTDBPair


def test_construct_where_clause_with_datetimes():
    start_datetime = datetime.datetime(2020, 1, 1, 0, 0, 0)
    end_datetime = datetime.datetime(2020, 2, 1, 0, 0, 0)
    timestamp_fields = ['created_at', 'updated_at']

    where_clause = _construct_where_clause_from_timerange(
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        timestamp_fields=timestamp_fields,
        stick_to_dates=False,  # !important
    )

    answ = " WHERE (created_at >= '2020-01-01 00:00:00' AND created_at < '2020-02-01 00:00:00')" \
           " OR (updated_at >= '2020-01-01 00:00:00' AND updated_at < '2020-02-01 00:00:00')"
    assert where_clause == answ, where_clause


def test_construct_where_clause_with_dates():
    start_datetime = datetime.datetime(2020, 1, 1)
    end_datetime = datetime.datetime(2020, 2, 1, )
    timestamp_fields = ['created_at', 'updated_at']

    where_clause = _construct_where_clause_from_timerange(
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        timestamp_fields=timestamp_fields,
        stick_to_dates=True,  # !important
    )

    answ = " WHERE (created_at >= '2020-01-01' AND created_at < '2020-02-01')" \
           " OR (updated_at >= '2020-01-01' AND updated_at < '2020-02-01')"
    assert where_clause == answ


def test_data_client_factory():
    DATABASES = {
        'db_key11': {
            'engine': 'mock_engine',
        },
        'db_key12': {
            'engine': 'mock_engine',
        },
    }
    factory = DataClientFactory(DATABASES)
    client = factory(db_key='db_key11')
    assert type(client) == DataClient


def test_elt_pair_factory():
    DATABASES = {
        'db_key11': {
            'engine': 'mock_engine',
        },
        'db_key12': {
            'engine': 'mock_engine',
        },
    }

    ELT_PAIRS = {
        'pair1': {
            'source': 'db_key11', 'target': 'db_key12'
        },
    }
    factory = ELTDBPairFactory(ELT_PAIRS, DATABASES)
    elt_pair = factory(pair_key='pair1')
    assert type(elt_pair) == ELTDBPair
