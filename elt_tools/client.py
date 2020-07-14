"""Generic Client for interacting with data sources."""
import datetime
import logging
import timeout_decorator
from timeout_decorator import TimeoutError
from decimal import Decimal
import math
from retrying import retry
from sqlalchemy import MetaData, Table
from sqlalchemy.inspection import inspect
from psycopg2.errors import SerializationFailure
from typing import Dict, Set, List, Tuple, Optional
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
        logging.warning(msg)
        return ''

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
        self._tables = {}

    @classmethod
    def from_settings(cls, database_settings: Dict, db_key):
        return cls(engine_from_settings(db_key, database_settings=database_settings))

    def table(self, table_name):
        """Introspect Table object from engine metadata."""
        if table_name in self._tables:
            logging.debug(f"Using cached definition of table {table_name}")
            return self._tables[table_name]
        else:
            logging.debug(f"Inspecting table {table_name}")
            t = Table(table_name, self.metadata, autoload=True)
            self._tables[table_name] = t
            logging.debug("Introspection done.")
            return t

    def insert_rows(self, rows, table=None, replace=None):
        """Insert rows into table."""
        if replace:
            self.engine.execute(f'TRUNCATE TABLE {table}')
        self.table_name = table
        self.engine.execute(self.table.insert(), rows)
        return self.construct_response(rows, table)

    def delete_rows(self, table_name, key_field, primary_keys=None ):
        if not primary_keys:
            logging.error("Pass in the primary keys to delete.")
            return
        def format_primary_key(key):
            if isinstance(key, int):
                return str(key)
            else:
                return f"'{str(key)}'"
        query = f"""
        DELETE {table_name} WHERE {key_field} IN ({','.join(map(format_primary_key, primary_keys))})
        """
        logging.info(query)
        self.query(query)

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

    @retry(stop_max_attempt_number=3, wait_fixed=5000)
    @timeout_decorator.timeout(60)
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
        unfiltered_count_query = f"""
        SELECT COUNT({field_name}) AS count FROM {table_name}
        """
        where_clause = _construct_where_clause_from_timerange(
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            timestamp_fields=timestamp_fields,
            stick_to_dates=stick_to_dates,
        )
        count_query = unfiltered_count_query + where_clause
        logging.debug("Count query is %s" % count_query)
        try:
            result = self.query(count_query)[0]['count']
        # sometimes with postgres dbs we encounter SerializationFailure when we query the slave and master
        # interrupts it. In this case, we sub-divide the query time range.
        except SerializationFailure as e:
            if where_clause:
                range_len = math.floor((end_datetime - start_datetime) / datetime.timedelta(hours=24))
                logging.info("Encountered exception with count query across %d days. Aggregating over single days. %s" % (
                    range_len, str(e)))
                range_split = [start_datetime + datetime.timedelta(days=n) for n in range(range_len)] + [end_datetime]
                result = 0
                for sub_start, sub_end in zip(range_split, range_split[1:]):
                    sub_count = self.count(
                        table_name,
                        field_name,
                        start_datetime=sub_start,
                        end_datetime=sub_end,
                        timestamp_fields=timestamp_fields,
                        stick_to_dates=stick_to_dates,
                    )
                    result += sub_count
            else:
                raise
        return result

    def get_primary_key(self, table_name: str) -> Optional[str]:
        """
        Inspect the table to find the primary key.
        """
        prim_key = self.table(table_name).primary_key
        if prim_key:
            key_cols = list(prim_key.columns)
            if len(key_cols) > 1:
                raise ValueError("Currently this toolset only supports sole primary keys."
                                 f"Found keys {key_cols} for {table_name}.")
            if key_cols:
                primary_key_field_name = key_cols[0].name
                logging.debug(f'Found primary key of {table_name} is {primary_key_field_name}.')
                return primary_key_field_name
            else:
                return None

    def get_all_tables(self) -> List[str]:
        """
        List all the table names present in the schema definition.
        """
        source_db_tables = []
        query = '''
              SELECT table_name
                FROM {bq_schema}INFORMATION_SCHEMA.TABLES
              ORDER BY table_name
            '''.format(
            bq_schema=self.engine.url.database + '.' if self.engine.name == 'bigquery' else ''
        )
        tables = self.fetch_rows(query)
        for table in tables:
            source_db_tables.append(table[0])
        return source_db_tables

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

    def _find_partition_expression(self, table_name):
        """
           Note: this currently only supports Google Biquery
        """
        partition_field_sql = f"""
            SELECT column_name, data_type
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE table_name = '{table_name}'
              AND is_partitioning_column = 'YES';
        """
        partition_field_result = self.query(partition_field_sql)
        partition_field_expr = None
        partition_field_name = None
        partition_field_type = None
        for row in partition_field_result:
            if partition_field_name:
                raise ValueError("Expecting one partition field, found multiple.")
            partition_field_name = row['column_name']
            partition_field_type = row['data_type']

        if partition_field_type == 'TIMESTAMP':
            partition_field_expr = f'DATE({partition_field_name})'
        elif partition_field_type == 'DATE':
            partition_field_expr = partition_field_name
        else:
            raise ValueError('Expected partition field to be either DATE OR TIMESTAMP.'
                             f' "{partition_field_type}" not supported. ')

        if partition_field_expr:
            return f'PARTITION BY {partition_field_expr}'
        else:
            return None

    def _find_cluster_expr(self, table_name):
        """
           Note: this currently only supports Google Biquery
        """

        cluster_fields_sql = f"""
           SELECT column_name
           FROM INFORMATION_SCHEMA.COLUMNS
           WHERE table_name = '{table_name}'
            AND clustering_ordinal_position IS NOT NULL
           ORDER BY clustering_ordinal_position ASC;
        """
        cluster_fields_result = self.query(cluster_fields_sql)
        cluster_field_exp = ','.join([r['column_name'] for r in cluster_fields_result])

        if cluster_field_exp:
            return 'CLUSTER BY ' + cluster_field_exp
        else:
            return None


    def remove_duplicate_keys_from_bigquery(self, table_name, key_field):
        """Remove any duplicate records when comparing primary keys.
           Note: this currently only supports Google Biquery
        """
        dups = self.find_duplicate_keys(table_name,  key_field)
        if dups:
            logging.info(f"Removing duplicates from {table_name}: {str(dups)}")
        else:
            logging.info(f"No duplicates found in {table_name}.")
            return

        partition_exp = self._find_partition_expression(table_name)
        cluster_exp = self._find_cluster_expr(table_name)

        sql = f"""
                    CREATE OR REPLACE TABLE {table_name}
                    {partition_exp if partition_exp else ''}
                    {cluster_exp if cluster_exp else ''}
                    AS
                    SELECT k.*
                    FROM (
                      SELECT ARRAY_AGG(row LIMIT 1)[OFFSET(0)] k 
                      FROM {table_name} row
                      GROUP BY {key_field}
                    )
                """
        logging.info(sql)
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

    def _find_orphans(
            self,
            table_name,
            key_field,
            start_datetime: datetime.datetime = None,
            end_datetime: datetime.datetime = None,
            timestamp_fields: List[str] = None,
            stick_to_dates: bool = False,
            **kwargs,
    ) -> Set:
        """
        Find orphaned records in target for which their source parents were deleted.
        Optionally pass in timestamp fields and time range to limit the amount of records
        to compare for orphans (for use on large tables).
        """
        orphans, _ = self._find_orphans_and_missing_in_target(
            table_name,
            key_field,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            timestamp_fields=timestamp_fields,
            stick_to_dates=stick_to_dates,
            **kwargs,
        )
        return orphans

    def find_orphans(
            self,
            table_name,
            key_field,
            start_datetime: datetime.datetime = None,
            end_datetime: datetime.datetime = None,
            timestamp_fields: List[str] = None,
            stick_to_dates: bool = False,
            **kwargs,
    ):
        orphans = self.find_by_recursive_date_range_bifurcation(
            table_name,
            key_field,
            self._find_orphans,
            bifurcation_against='target',
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            timestamp_fields=timestamp_fields,
            stick_to_dates=stick_to_dates,
            **kwargs,
        )
        return orphans

    def remove_orphans_from_target(
            self,
            table_name,
            key_field,
            start_datetime: datetime.datetime = None,
            end_datetime: datetime.datetime = None,
            timestamp_fields: List[str] = None,
            stick_to_dates: bool = False,
            **kwargs,
    ):
        orphans = self.find_orphans(
            table_name,
            key_field,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            timestamp_fields=timestamp_fields,
            stick_to_dates=stick_to_dates,
            **kwargs,
        )
        self.target.delete_rows(table_name, key_field, primary_keys=orphans)
        return orphans

    def find_by_recursive_date_range_bifurcation(
            self,
            table_name,
            key_field,
            find_func,
            bifurcation_against='target',
            start_datetime: datetime.datetime = None,
            end_datetime: datetime.datetime = None,
            timestamp_fields: List[str] = None,
            stick_to_dates: bool = False,
            thres=1000000,
            min_segment_size=datetime.timedelta(days=1),
            dry_run=False,
            skip_based_on_count=False,
    ) -> Set:
        """
        Do binary search on table by recursively bifurcating the date range
        until the number of records in that range drops below the threshold.
        Once, below the threshold, apply the specified find function.
        :param table_name:
        :param key_field:
        :param find_func: Find function to apply. Must take same params as this function, except `func` param.
                     Must return a set of primary keys of the records it found.
        :param bifurcation_against: Either 'source' or 'target'. Choose which DB in the pair you want to split the data
                                    range against.
        :param start_datetime:
        :param end_datetime:
        :param timestamp_fields:
        :param stick_to_dates:
        :param thres:
        :param min_segment_size:
        :param dry_run:
        :return: Set of primary keys of whatever is found
        """
        if not timestamp_fields:
            logging.warning("For a more efficient search, specify the timestamp field(s).")
            return self._find_missing(
                table_name,
                key_field,
                stick_to_dates=stick_to_dates,
                skip_based_on_count=skip_based_on_count,
            )

        bifurcation_against_lookup = {
            'source': self.source,
            'target': self.target,
        }
        # If time range is not set, fetch it from the target database
        if not start_datetime:
            query = """
            SELECT {select_stmt}
            FROM {table_name}
            """.format(
                select_stmt=', '.join(map(lambda x: 'MIN({0}) AS {0} '.format(x), timestamp_fields)),
                table_name=table_name,
            )
            result = bifurcation_against_lookup[bifurcation_against].query(query)
            start_datetime = min(result[0].values())
        if not end_datetime:
            query = """
            SELECT {select_stmt}
            FROM {table_name}
            """.format(
                select_stmt=', '.join(map(lambda x: 'MAX({0}) AS {0} '.format(x), timestamp_fields)),
                table_name=table_name,
            )
            result = bifurcation_against_lookup[bifurcation_against].query(query)
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

        find_result = set()
        count1 = count2 = 0
        start, halfway, end = bifurcate_time_range(start_datetime, end_datetime)
        logging.debug(f"Start date is : {start}")
        logging.debug(f"Halfway date is : {halfway}")
        logging.debug(f"End date is : {end}")

        if start != halfway:
            try:
                count1 = bifurcation_against_lookup[bifurcation_against].count(
                    table_name,
                    field_name=key_field,
                    start_datetime=start,
                    end_datetime=halfway,
                    timestamp_fields=timestamp_fields,
                    stick_to_dates=stick_to_dates,
                )
                logging.debug(f"Count of range 1 is {count1}")
            except TimeoutError:
                # if count times out, assume it's larger than the threshold
                count1 = thres + 1
                logging.debug(f"Count 1 timed out, assuming greater than threshold of {thres}.")

        if end != halfway:
            try:
                count2 = bifurcation_against_lookup[bifurcation_against].count(
                    table_name,
                    field_name=key_field,
                    start_datetime=halfway,
                    end_datetime=end,
                    timestamp_fields=timestamp_fields,
                    stick_to_dates=stick_to_dates,
                )
                logging.debug(f"Count of range 2 is {count2}")
            except TimeoutError:
                # if count times out, assume it's larger than the threshold
                count2 = thres + 1
                logging.debug(f"Count 2 timed out, assuming greater than threshold of {thres}.")


        # exit conditions
        if start == halfway or end == halfway or (halfway - start) < min_segment_size:
            return find_func(
                table_name,
                key_field,
                start_datetime=start,
                end_datetime=end,
                timestamp_fields=timestamp_fields,
                stick_to_dates=stick_to_dates,
                dry_run=dry_run,
                skip_based_on_count=skip_based_on_count,
            )
        if count1 == 0 and count2 == 0:
            return set()
        if count1 < thres and count2 < thres:
            return find_func(
                table_name,
                key_field,
                start_datetime=start,
                end_datetime=halfway,
                timestamp_fields=timestamp_fields,
                stick_to_dates=stick_to_dates,
                dry_run=dry_run,
                skip_based_on_count=skip_based_on_count,
            ) | find_func(
                table_name,
                key_field,
                start_datetime=halfway,
                end_datetime=end,
                timestamp_fields=timestamp_fields,
                stick_to_dates=stick_to_dates,
                dry_run=dry_run,
                skip_based_on_count=skip_based_on_count,
            )

        # recursion conditions
        if count1 >= thres or count2 >= thres:
            find_result |= self.find_by_recursive_date_range_bifurcation(
                table_name,
                key_field,
                find_func=find_func,
                start_datetime=start,
                end_datetime=halfway,
                timestamp_fields=timestamp_fields,
                stick_to_dates=stick_to_dates,
                thres=thres,
                skip_based_on_count=skip_based_on_count,
            ) | self.find_by_recursive_date_range_bifurcation(
                table_name,
                key_field,
                find_func=find_func,
                start_datetime=halfway,
                end_datetime=end,
                timestamp_fields=timestamp_fields,
                stick_to_dates=stick_to_dates,
                thres=thres,
                skip_based_on_count=skip_based_on_count,
            )

        return find_result

    def _find_orphans_and_missing_in_target(
            self,
            table_name,
            key_field,
            start_datetime: datetime.datetime = None,
            end_datetime: datetime.datetime = None,
            timestamp_fields: List[str] = None,
            stick_to_dates: bool = False,
            skip_based_on_count: bool = True,
            **kwargs,
    ) -> Tuple[Set, Set]:
        """
        Find orphaned records and missing records in target compared to the source.
        Optionally pass in timestamp fields and time range to limit the amount of records
        to compare for missing records.
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
        logging.debug("Id lookup for find query: %s" % all_ids_query)

        # First compare counts on limited date range to skip id comparison if no difference
        if where_clause and skip_based_on_count:
            count_diff = self.compare_counts(
                table_name,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                timestamp_fields=timestamp_fields,
                stick_to_dates=stick_to_dates,
            )
            if count_diff == 0:
                return set(), set()

        def format_id(row):
            id = row['id']
            if isinstance(id, int):
                return id
            else:
                return str(id)  # cast to str if UUID

        target_rows = self.target.fetch_rows(all_ids_query)
        target_ids = set(map(format_id, target_rows))

        source_rows = self.source.fetch_rows(all_ids_query)
        source_ids = set(map(format_id, source_rows))

        missing = source_ids - target_ids
        orphans = target_ids - source_ids
        return orphans, missing

    def _find_missing(
            self,
            table_name,
            key_field,
            start_datetime: datetime.datetime = None,
            end_datetime: datetime.datetime = None,
            timestamp_fields: List[str] = None,
            stick_to_dates: bool = False,
            **kwargs,
    ) -> Set:
        """
        Find missing records in target for which their source parents were deleted.
        Optionally pass in timestamp fields and time range to limit the amount of records
        to compare for missing records.
        """
        _, missing = self._find_orphans_and_missing_in_target(
            table_name,
            key_field,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            timestamp_fields=timestamp_fields,
            stick_to_dates=stick_to_dates,
            **kwargs,
        )
        return missing

    def find_missing(
            self,
            table_name,
            key_field,
            start_datetime: Optional[datetime.datetime] = None,
            end_datetime: Optional[datetime.datetime] = None,
            timestamp_fields: List[str] = None,
            stick_to_dates: Optional[bool] = False,
            **kwargs,
    ) -> Set:
        """
        Find primary keys of missing records in target for which their source parents were deleted.
        Optionally pass in timestamp fields and time range to limit the amount of records
        to compare for missing records.
        """
        missing = self.find_by_recursive_date_range_bifurcation(
            table_name,
            key_field,
            self._find_missing,
            bifurcation_against='source',
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            timestamp_fields=timestamp_fields,
            stick_to_dates=stick_to_dates,
            **kwargs,
        )
        return missing

    def get_common_tables(self):
        source_tables = set(self.source.get_all_tables())
        target_tables = set(self.target.get_all_tables())
        return sorted(source_tables.intersection(target_tables))

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
