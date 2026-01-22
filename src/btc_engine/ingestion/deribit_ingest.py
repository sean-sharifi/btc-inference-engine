"""Deribit data ingestion module"""

from datetime import datetime, timedelta
from typing import Optional
import pandas as pd

from btc_engine.connectors.deribit import create_deribit_connector
from btc_engine.database.client import db_client
from btc_engine.utils.logging_config import logger


class DeribitIngestor:
    """Ingest Deribit options and futures data into database"""
    
    def __init__(self):
        """Initialize Deribit ingestor"""
        self.connector = create_deribit_connector()
    
    def ingest_instruments(self) -> int:
        """Ingest current instruments list
        
        Returns:
            Number of instruments ingested
        """
        logger.info("Ingesting Deribit instruments")
        
        instruments = self.connector.get_instruments(expired=False)
        timestamp = datetime.now()
        
        records = []
        for inst in instruments:
            record = {
                "timestamp": timestamp,
                "instrument_name": inst.get("instrument_name"),
                "strike": inst.get("strike"),
                "expiration_timestamp": datetime.fromtimestamp(inst.get("expiration_timestamp") / 1000),
                "option_type": inst.get("option_type"),
                "creation_timestamp": datetime.fromtimestamp(inst.get("creation_timestamp") / 1000),
                "is_active": inst.get("is_active", True),
                "min_trade_amount": inst.get("min_trade_amount"),
                "tick_size": inst.get("tick_size"),
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Insert into database
        db_client.insert_dataframe("raw_deribit_instruments", df, if_exists="append")
        
        logger.info(f"Ingested {len(df)} instruments")
        return len(df)
    
    def ingest_ticker_snapshot(self) -> int:
        """Ingest current ticker snapshot for all options
        
        Returns:
            Number of tickers ingested
        """
        logger.info("Ingesting Deribit ticker snapshot")
        
        df = self.connector.snapshot_iv_surface()
        
        # Insert into database
        db_client.insert_dataframe("raw_deribit_ticker_snapshots", df, if_exists="append")
        
        logger.info(f"Ingested {len(df)} ticker snapshots")
        return len(df)
    
    def ingest_funding_rate(self) -> int:
        """Ingest current funding rate
        
        Returns:
            Number of records ingested
        """
        logger.info("Ingesting Deribit funding rate")
        
        funding_data = self.connector.get_funding_rate()
        
        df = pd.DataFrame([{
            "timestamp": datetime.fromtimestamp(funding_data["timestamp"] / 1000) if funding_data.get("timestamp") else datetime.now(),
            "instrument_name": funding_data["instrument_name"],
            "funding_rate": funding_data.get("current_funding"),
            "funding_8h": funding_data.get("funding_8h"),
            "index_price": funding_data.get("index_price"),
            "mark_price": funding_data.get("mark_price"),
            "open_interest": funding_data.get("open_interest"),
        }])
        
        # Insert into database
        db_client.insert_dataframe("raw_deribit_funding", df, if_exists="append")
        
        logger.info(f"Ingested funding rate")
        return 1
    
    def ingest_full_snapshot(self) -> dict:
        """Ingest full snapshot: instruments, tickers, and funding
        
        Returns:
            Dictionary with counts of ingested records
        """
        logger.info("Starting full Deribit snapshot ingestion")
        
        result = {
            "instruments": self.ingest_instruments(),
            "tickers": self.ingest_ticker_snapshot(),
            "funding": self.ingest_funding_rate(),
            "timestamp": datetime.now()
        }
        
        logger.info(f"Full snapshot complete: {result}")
        return result
    
    def close(self):
        """Close connector"""
        self.connector.close()


def run_deribit_ingestion(full_snapshot: bool = True) -> dict:
    """Run Deribit data ingestion
    
    Args:
        full_snapshot: If True, ingest instruments + tickers + funding. Otherwise only tickers.
        
    Returns:
        Dictionary with ingestion results
    """
    ingestor = DeribitIngestor()
    
    try:
        if full_snapshot:
            return ingestor.ingest_full_snapshot()
        else:
            return {
                "tickers": ingestor.ingest_ticker_snapshot(),
                "timestamp": datetime.now()
            }
    finally:
        ingestor.close()
