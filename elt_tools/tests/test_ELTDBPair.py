import datetime
from elt_tools.client import ELTDBPair


def test_construct_where_clause_with_datetimes():
    start_datetime = datetime.datetime(2020, 1, 1, 0, 0, 0)
    end_datetime = datetime.datetime(2020, 2, 1, 0, 0, 0)
    timestamp_fields = ['created_at', 'updated_at']

    where_clause = ELTDBPair._construct_where_clause_from_timerange(
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        timestamp_fields=timestamp_fields,
        stick_to_dates=False,  # !important
    )

    answ = " WHERE created_at >= '2020-01-01 00:00:00' AND updated_at >= '2020-01-01 00:00:00'" \
           " AND created_at < '2020-02-01 00:00:00' AND updated_at < '2020-02-01 00:00:00'"
    assert where_clause == answ


def test_construct_where_clause_with_dates():
    start_datetime = datetime.datetime(2020, 1, 1)
    end_datetime = datetime.datetime(2020, 2, 1,)
    timestamp_fields = ['created_at', 'updated_at']

    where_clause = ELTDBPair._construct_where_clause_from_timerange(
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        timestamp_fields=timestamp_fields,
        stick_to_dates=True,  # !important
    )

    answ = " WHERE created_at >= '2020-01-01' AND updated_at >= '2020-01-01'" \
           " AND created_at < '2020-02-01' AND updated_at < '2020-02-01'"
    assert where_clause == answ
