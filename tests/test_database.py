"""Test database schema and client"""

import pytest
from btc_engine.database.schema import create_tables, get_table_info
from btc_engine.database.client import DatabaseClient


def test_create_tables(temp_db):
    """Test table creation"""
    create_tables(temp_db)
    
    table_info = get_table_info(temp_db)
    
    assert len(table_info) > 0
    assert 'raw_deribit_instruments' in table_info
    assert 'raw_glassnode_metrics' in table_info
    assert 'features_options_surface' in table_info


def test_database_client(temp_db):
    """Test database client operations"""
    create_tables(temp_db)
    
    client = DatabaseClient(temp_db)
    
    # Test table exists
    assert client.table_exists('raw_deribit_instruments')
    assert not client.table_exists('nonexistent_table')
    
    # Test row count
    count = client.get_row_count('raw_deribit_instruments')
    assert count == 0
