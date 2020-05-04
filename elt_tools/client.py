"""Generic Client for interacting with data sources."""
import datetime
import logging
from sqlalchemy import MetaData, Table
from typing import Dict, Set, List
from elt_tools.engines import engine_from_settings


def _construct_where_clause_from_timerange(
        start_datetime: datetime.datetime = None,
        end_datetime: datetime.datetime = None,
        timestamp_fields: List[str] = None,
        stick_to_dates: bool = False
):
    where_clause = ""

    if stick_to_dates and start_datetime == end_datetime:
        msg = "The date range for dates is inclusive of the start date and exclusive of the end date." \
              "Since your start and end date range is the same, your query will be meaningless."
        raise ValueError(msg)

    if (not timestamp_fields and start_datetime) or (not timestamp_fields and end_datetime):
        msg = "You've passed in a time range, but no timestamp field names."
        print(msg)
        raise ValueError(msg)

    if stick_to_dates:
        fmt_string = "%Y-%m-%d"
    else:
        fmt_string = "%Y-%m-%d %H:%M:%S"

    if start_datetime:
        start_datetime = start_datetime.strftime(fmt_string)
    if end_datetime:
        end_datetime = end_datetime.strftime(fmt_string)

    if timestamp_fields and start_datetime and end_datetime:
        where_clause += " WHERE " + " OR ".join([
            f"({timestamp_field} >= '{start_datetime}' AND {timestamp_field} < '{end_datetime}')"
            for timestamp_field in timestamp_fields
        ])
        return where_clause

    if timestamp_fields and start_datetime:
        where_clause += " WHERE " + " AND ".join([
            f"{timestamp_field} >= '{start_datetime}'"
            for timestamp_field in timestamp_fields
        ])
    if timestamp_fields and end_datetime:
        where_clause += " AND " + " AND ".join([
            f"{timestamp_field} < '{end_datetime}'"
            for timestamp_field in timestamp_fields
        ])
    return where_clause


class DataClient:

    @classmethod
    def update_settings(cls, item_name):
        """

        :param item_name: Corresponds to settings class-attribute name
        :return:
        """

    def __init__(self, engine):
        self.engine = engine
        self.metadata = MetaData(bind=self.engine)
        self.table_name = None

    @classmethod
    def from_settings(cls, database_settings: Dict, db_key):
        return cls(engine_from_settings(db_key, database_settings=database_settings))

    @property
    def table(self):
        if self.table_name:
            return Table(self.table_name, self.metadata, autoload=True)
        return None

    def insert_rows(self, rows, table=None, replace=None):
        """Insert rows into table."""
        if replace:
            self.engine.execute(f'TRUNCATE TABLE {table}')
        self.table_name = table
        self.engine.execute(self.table.insert(), rows)
        return self.construct_response(rows, table)

    def fetch_rows(self, query):
        """Fetch all rows via query."""
        rows = self.engine.execute(query).fetchall()
        return rows

    def query(self, query):
        return [dict(r) for r in self.fetch_rows(query)]

    @staticmethod
    def construct_response(rows, table):
        """Summarize results of an executed query."""
        columns = rows[0].keys()
        column_names = ", ".join(columns)
        num_rows = len(rows)
        return f'Inserted {num_rows} rows into `{table}` with {len(columns)} columns: {column_names}'

    def count(
            self,
            table_name,
            field_name=None,
            start_datetime: datetime.datetime = None,
            end_datetime: datetime.datetime = None,
            timestamp_fields: List[str] = None,
            stick_to_dates: bool = False
    ) -> int:
        """
        Optionally pass in timestamp fields and time range to limit the query_range.
        """
        if not field_name:
            field_name = "*"
        count_query = f"""
        SELECT COUNT({field_name}) AS count FROM {table_name}
        """
        where_clause = _construct_where_clause_from_timerange(
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            timestamp_fields=timestamp_fields,
            stick_to_dates=stick_to_dates,
        )
        count_query += where_clause
        logging.debug("Count query is %s" % count_query)
        result = self.query(count_query)[0]['count']
        return result

    def find_duplicate_keys(self, table_name, key_field):
        """Find if a table has duplicates by a certain column, if so return all the instances that
        have duplicates together with their counts."""
        query = f"""
        SELECT {key_field}, COUNT({key_field}) as count
        FROM {table_name}
        GROUP BY 1
        HAVING count > 1;
        """
        return self.fetch_rows(query)

    def remove_duplicate_keys(self, table_name, key_field):
        """Remove any duplicate records when comparing primary keys."""
        dups = self.find_duplicate_keys(table_name,  key_field)
        if dups:
            logging.info(f"Removing duplicates from {table_name}: {str(dups)}")
        else:
            logging.info(f"No duplicates found in {table_name}.")
            return

        sql = f"""
            DELETE FROM {table_name}
            WHERE
                {key_field} IN (
                SELECT
                    {key_field}
                FROM (
                    SELECT
                        {key_field},
                        ROW_NUMBER() OVER (
                            PARTITION BY {key_field}
                            ORDER BY {key_field}) AS row_num
                    FROM
                        {table_name}
                ) t
                WHERE row_num > 1
            );
        """
        self.query(sql)
        logging.info("Duplicates removed.")


class DataClientFactory:

    def __init__(self, database_settings):
        self.database_settings = database_settings

    def __call__(self, db_key=None):
        return DataClient.from_settings(
            self.database_settings,
            db_key,
        )


class ELTDBPair:

    def __init__(self, name: str, source: DataClient, target: DataClient):
        self.name = name
        self.source = source
        self.target = target

    @classmethod
    def from_settings(cls, elt_pair_settings, database_settings, pair_key, name=None):
        if not name:
            name = pair_key
        if not pair_key:
            pair_key = name
        source_target_settings = elt_pair_settings[pair_key]
        source_client = DataClient.from_settings(database_settings, source_target_settings['source'])
        target_client = DataClient.from_settings(database_settings, source_target_settings['target'])
        return cls(name, source_client, target_client)

    def __repr__(self):
        return self.name

    def compare_counts(
            self,
            table_name,
            field_name=None,
            start_datetime: datetime.datetime = None,
            end_datetime: datetime.datetime = None,
            timestamp_fields: List[str] = None,
            stick_to_dates: bool = False
    ) -> int:
        return self.target.count(
            table_name,
            field_name=field_name,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            timestamp_fields=timestamp_fields,
            stick_to_dates=stick_to_dates,
        ) - self.source.count(
            table_name,
            field_name=field_name,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            timestamp_fields=timestamp_fields,
            stick_to_dates=stick_to_dates,
        )

    def find_orphans(
            self,
            table_name,
            key_field,
            start_datetime: datetime.datetime = None,
            end_datetime: datetime.datetime = None,
            timestamp_fields: List[str] = None,
            stick_to_dates: bool = False
    ) -> Set:
        """
        Find orphaned records in BQ for which their source parents were deleted.
        Optionally pass in timestamp fields and time range to limit the amount of records
        to compare for orphans (for use on large tables).
        """
        all_ids_query = f"""
            SELECT {key_field} AS id FROM {table_name}
        """

        where_clause = _construct_where_clause_from_timerange(
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            timestamp_fields=timestamp_fields,
            stick_to_dates=stick_to_dates,
        )
        all_ids_query += where_clause
        logging.debug("Id lookup for orphan query: %s" % all_ids_query)

        # First compare counts on limited date range to skip id comparison if no difference
        if where_clause:
            count_diff = self.compare_counts(
                table_name,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                timestamp_fields=timestamp_fields,
                stick_to_dates=stick_to_dates,
            )
            if count_diff == 0:
                return set()

        def format_id(row):
            id = row['id']
            if isinstance(id, int):
                return id
            else:
                return str(id)  # cast to str if UUID

        rows = self.target.fetch_rows(all_ids_query)
        target_ids = set(map(format_id, rows))

        rows = self.source.fetch_rows(all_ids_query)
        source_ids = set(map(format_id, rows))

        orphans = target_ids - source_ids
        return orphans

    def remove_orphans_from_target(
            self,
            table_name,
            key_field,
            start_datetime: datetime.datetime = None,
            end_datetime: datetime.datetime = None,
            timestamp_fields: List[str] = None,
            stick_to_dates: bool = False,
            dry_run: bool = False,
    ) -> int:
        orphans = self.find_orphans(
            table_name,
            key_field,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            timestamp_fields=timestamp_fields,
            stick_to_dates=stick_to_dates
        )
        num_orphans = len(orphans)

        def format_primary_key(k):
            if isinstance(k, int):
                return str(k)
            else:
                return "'%s'" % str(k)

        orphan_ids_csv = ','.join(map(format_primary_key, orphans))

        if not orphan_ids_csv:
            logging.info("No orphans found for table %s" % table_name)
        else:
            logging.info("Found %d orphaned records in target %s." % (num_orphans, table_name))
            delete_query = f"""
            DELETE {table_name}
            WHERE {key_field} IN ({orphan_ids_csv})
            """
            logging.info(delete_query)
            if not dry_run:
                self.target.query(delete_query)

        return num_orphans

    def remove_orphans_from_target_with_binary_search(
            self,
            table_name,
            key_field,
            start_datetime: datetime.datetime = None,
            end_datetime: datetime.datetime = None,
            timestamp_fields: List[str] = None,
            stick_to_dates: bool = False,
            thres=10000,
            min_segment_size=datetime.timedelta(seconds=10),
            dry_run=False,
    ):
        """
        Do binary search on table by recursively bifurcating the date range
        until the number of records in that range drops below the threshold.
        Once, below the threshold, find and remove the orphans for that segment.
        :return: Total number of records removed.
        """
        # If time range is not set, fetch it from the target database
        if not start_datetime:
            query = """
            SELECT {select_stmt}
            FROM {table_name}
            """.format(
                select_stmt=', '.join(map(lambda x: 'MIN({0}) AS {0} '.format(x), timestamp_fields)),
                table_name=table_name,
            )
            result = self.target.query(query)
            start_datetime = min(result[0].values())
        if not end_datetime:
            query = """
            SELECT {select_stmt}
            FROM {table_name}
            """.format(
                select_stmt=', '.join(map(lambda x: 'MAX({0}) AS {0} '.format(x), timestamp_fields)),
                table_name=table_name,
            )
            result = self.target.query(query)
            end_datetime = max(result[0].values())

        if stick_to_dates:
            if start_datetime:
                start_datetime = start_datetime.date()
            if end_datetime:
                end_datetime = end_datetime.date()

        def avg_datetime(start, end):
            return start + (end - start) / 2

        def bifurcate_time_range(start, end):
            halfway = avg_datetime(start, end)
            return start, halfway, end

        removed_count = 0
        count1 = count2 = 0
        start, halfway, end = bifurcate_time_range(start_datetime, end_datetime)
        logging.debug(f"Start date is : {start}")
        logging.debug(f"Halfway date is : {halfway}")
        logging.debug(f"End date is : {end}")

        if start != halfway:
            count1 = self.target.count(
                table_name,
                field_name=key_field,
                start_datetime=start,
                end_datetime=halfway,
                timestamp_fields=timestamp_fields,
                stick_to_dates=stick_to_dates,
            )
        if end != halfway:
            count2 = self.target.count(
                table_name,
                field_name=key_field,
                start_datetime=halfway,
                end_datetime=end,
                timestamp_fields=timestamp_fields,
                stick_to_dates=stick_to_dates,
            )

        logging.debug(f"Count of range 1 is {count1}")
        logging.debug(f"Count of range 2 is {count2}")

        # exit conditions
        if start == halfway or end == halfway or (halfway - start) < min_segment_size:
            return self.remove_orphans_from_target(
                table_name,
                key_field,
                start_datetime=start,
                end_datetime=end,
                timestamp_fields=timestamp_fields,
                stick_to_dates=stick_to_dates,
                dry_run=dry_run,
            )
        if count1 == 0 and count2 == 0:
            return 0
        if count1 < thres and count2 < thres:
            return self.remove_orphans_from_target(
                table_name,
                key_field,
                start_datetime=start,
                end_datetime=halfway,
                timestamp_fields=timestamp_fields,
                stick_to_dates=stick_to_dates,
                dry_run=dry_run,
            ) + self.remove_orphans_from_target(
                table_name,
                key_field,
                start_datetime=halfway,
                end_datetime=end,
                timestamp_fields=timestamp_fields,
                stick_to_dates=stick_to_dates,
                dry_run=dry_run,
            )

        # recursion conditions
        if count1 >= thres or count2 >= thres:
            removed_count += self.remove_orphans_from_target_with_binary_search(
                table_name,
                key_field,
                start_datetime=start,
                end_datetime=halfway,
                timestamp_fields=timestamp_fields,
                stick_to_dates=stick_to_dates,
                thres=thres,
            ) + self.remove_orphans_from_target_with_binary_search(
                table_name,
                key_field,
                start_datetime=halfway,
                end_datetime=end,
                timestamp_fields=timestamp_fields,
                stick_to_dates=stick_to_dates,
                thres=thres,
            )

        return removed_count

    def find_missing_in_target(self):
        """
        This needs to incorporate some form of time lag in the query on both source and target
        to compensate for ELT latency.
        """
        pass

    def list_common_tables(self):
        pass

    def fill_missing_target_records(self):
        pass

    def remove_duplicates_from_target(self):
        pass


class ELTDBPairFactory:

    def __init__(self, elt_pair_settings, database_settings):
        self.elt_pair_settings = elt_pair_settings
        self.database_settings = database_settings

    def __call__(self, name=None, pair_key=None):
        return ELTDBPair.from_settings(
            self.elt_pair_settings,
            self.database_settings,
            pair_key,
            name=name,
        )
