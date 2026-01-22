"""Glassnode data ingestion module"""

from datetime import datetime, timedelta
from typing import Optional
import pandas as pd

from btc_engine.connectors.glassnode import create_glassnode_connector
from btc_engine.database.client import db_client
from btc_engine.utils.logging_config import logger


class GlassnodeIngestor:
    """Ingest Glassnode onchain data into database"""
    
    def __init__(self):
        """Initialize Glassnode ingestor"""
        self.connector = create_glassnode_connector()
    
    def ingest_metrics(
        self,
        since: Optional[int] = None,
        until: Optional[int] = None,
        incremental: bool = True
    ) -> int:
        """Ingest all configured Glassnode metrics
        
        Args:
            since: Start timestamp. If None and incremental=True, uses last timestamp from DB
            until: End timestamp. If None, uses current time
            incremental: If True, only fetch data since last ingestion
            
        Returns:
            Number of records ingested
        """
        logger.info("Ingesting Glassnode metrics")
        
        # Determine time range
        if until is None:
            until = int(datetime.now().timestamp())
        
        if since is None and incremental:
            # Get last timestamp from database
            last_ts = db_client.get_latest_timestamp("raw_glassnode_metrics")
            if last_ts:
                since = int(last_ts.timestamp())
                logger.info(f"Incremental mode: fetching data from {last_ts}")
            else:
                # No data yet, fetch last 90 days
                since = int((datetime.now() - timedelta(days=90)).timestamp())
                logger.info("No existing data, fetching last 90 days")
        elif since is None:
            # Full refresh, get last 90 days
            since = int((datetime.now() - timedelta(days=90)).timestamp())
        
        # Fetch all metrics
        df = self.connector.get_all_onchain_metrics(since=since, until=until)
        
        if len(df) == 0:
            logger.warning("No new data fetched from Glassnode")
            return 0
        
        # Add resolution column
        df['resolution'] = self.connector.resolution
        
        # Reorder columns to match table schema: timestamp, metric_name, value, symbol, resolution
        df = df[['timestamp', 'metric_name', 'value', 'symbol', 'resolution']]
        
        # Insert into database
        db_client.insert_dataframe("raw_glassnode_metrics", df, if_exists="append")
        
        logger.info(f"Ingested {len(df)} Glassnode metric records")
        return len(df)
    
    def close(self):
        """Close connector"""
        self.connector.close()


def run_glassnode_ingestion(
    since: Optional[int] = None,
    until: Optional[int] = None,
    incremental: bool = True
) -> dict:
    """Run Glassnode data ingestion
    
    Args:
        since: Start timestamp
        until: End timestamp
        incremental: Use incremental mode
        
    Returns:
        Dictionary with ingestion results
    """
    ingestor = GlassnodeIngestor()
    
    try:
        count = ingestor.ingest_metrics(since=since, until=until, incremental=incremental)
        return {
            "records": count,
            "timestamp": datetime.now()
        }
    finally:
        ingestor.close()
