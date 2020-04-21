"""Generic Client for interacting with data sources."""
from sqlalchemy import MetaData, Table
from typing import Dict

from elt_tools.engines import engine_from_settings
from elt_tools.settings import ELT_PAIRS, DATABASES


class DataClient:

    # Override this if you want
    # to pass in your settings.
    databases = DATABASES

    def __init__(self, engine):
        self.engine = engine
        self.metadata = MetaData(bind=self.engine)
        self.table_name = None

    @classmethod
    def from_settings(cls, db_key, databases:Dict=None):
        if databases:
            db_settings = databases
        else:
            db_settings = cls.databases
        return cls(engine_from_settings(db_key, databases=db_settings))

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


class ELTDBPair:

    # Override these if you want
    # to pass in your settings.
    elt_pairs = ELT_PAIRS
    databases = DATABASES


    def __init__(self, name:str, source: DataClient, target: DataClient):
        self.name = name
        self.source = source
        self.target = target

    @classmethod
    def from_settings(cls, name=None, db_key=None):
        if not name:
            name = db_key
        if not db_key:
            db_key = name
        source_target_settings = cls.elt_pairs[db_key]
        source_client = DataClient.from_settings(source_target_settings['source'], databases=cls.databases)
        target_client = DataClient.from_settings(source_target_settings['target'], databases=cls.databases)
        return cls(name, source_client, target_client)

    def __repr__(self):
        return self.name

    def compare_counts(self, table_name):
        count_query = f"""
        SELECT COUNT(*) AS count FROM {table_name};
        """
        return self.target.query(count_query)[0]['count'] - self.source.query(count_query)[0]['count']

    def find_orphans(self, table_name, key_field):
        """Find orphaned records in BQ for which their source parents were deleted."""
        all_ids_query = f"""
            SELECT {key_field} FROM {table_name};
        """
        rows = self.target.fetch_rows(all_ids_query)
        target_ids = {row[0] for row in rows}

        rows = self.source.fetch_rows(all_ids_query)
        source_ids = {str(row[0]) for row in rows}  # cast to string in case UUID

        orphans = target_ids - source_ids
        return orphans

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