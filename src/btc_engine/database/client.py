"""DuckDB connection manager and database client"""

from typing import Optional, Any, List, Dict
from contextlib import contextmanager
import duckdb
from pathlib import Path

from btc_engine.utils.config_loader import settings
from btc_engine.utils.logging_config import logger


class DatabaseClient:
    """DuckDB database client with connection management"""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database client
        
        Args:
            db_path: Path to DuckDB database file. If None, uses settings.database_path
        """
        self.db_path = db_path or settings.database_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"DatabaseClient initialized with path: {self.db_path}")
    
    @contextmanager
    def get_connection(self, read_only: bool = False):
        """Context manager for database connections
        
        Args:
            read_only: If True, open connection in read-only mode
            
        Yields:
            DuckDB connection
        """
        conn = duckdb.connect(self.db_path, read_only=read_only)
        try:
            yield conn
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: Optional[tuple] = None, 
                     read_only: bool = True) -> List[tuple]:
        """Execute a query and return results
        
        Args:
            query: SQL query string
            params: Optional query parameters
            read_only: If True, use read-only connection
            
        Returns:
            List of result tuples
        """
        with self.get_connection(read_only=read_only) as conn:
            if params:
                result = conn.execute(query, params).fetchall()
            else:
                result = conn.execute(query).fetchall()
        return result
    
    def execute_many(self, query: str, data: List[tuple]) -> None:
        """Execute a query with multiple parameter sets
        
        Args:
            query: SQL query string
            data: List of parameter tuples
        """
        with self.get_connection(read_only=False) as conn:
            conn.executemany(query, data)
            conn.commit()
    
    def insert_dataframe(self, table_name: str, df: Any, 
                        if_exists: str = "append") -> None:
        """Insert pandas DataFrame into table
        
        Args:
            table_name: Name of target table
            df: Pandas DataFrame to insert
            if_exists: What to do if table exists ('append', 'replace', 'fail')
        """
        with self.get_connection(read_only=False) as conn:
            # DuckDB can directly ingest pandas DataFrames
            if if_exists == "replace":
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")
            conn.commit()
            
        logger.debug(f"Inserted {len(df)} rows into {table_name}")
    
    def query_to_dataframe(self, query: str, params: Optional[tuple] = None) -> Any:
        """Execute query and return results as pandas DataFrame
        
        Args:
            query: SQL query string
            params: Optional query parameters
            
        Returns:
            Pandas DataFrame with results
        """
        with self.get_connection(read_only=True) as conn:
            if params:
                df = conn.execute(query, params).df()
            else:
                df = conn.execute(query).df()
        return df
    
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists in database
        
        Args:
            table_name: Name of table to check
            
        Returns:
            True if table exists, False otherwise
        """
        with self.get_connection(read_only=True) as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                (table_name,)
            ).fetchone()
            return result[0] > 0 if result else False
    
    def get_row_count(self, table_name: str) -> int:
        """Get number of rows in table
        
        Args:
            table_name: Name of table
            
        Returns:
            Number of rows
        """
        with self.get_connection(read_only=True) as conn:
            result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            return result[0] if result else 0
    
    def get_latest_timestamp(self, table_name: str, 
                            timestamp_col: str = "timestamp") -> Optional[Any]:
        """Get the latest timestamp from a table
        
        Args:
            table_name: Name of table
            timestamp_col: Name of timestamp column
            
        Returns:
            Latest timestamp or None if table is empty
        """
        try:
            with self.get_connection(read_only=True) as conn:
                result = conn.execute(
                    f"SELECT MAX({timestamp_col}) FROM {table_name}"
                ).fetchone()
                return result[0] if result and result[0] else None
        except Exception as e:
            logger.warning(f"Could not get latest timestamp from {table_name}: {e}")
            return None


# Singleton database client
db_client = DatabaseClient()
