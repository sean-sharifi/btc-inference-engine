"""Database migration and initialization"""

from btc_engine.database.schema import create_tables, get_table_info
from btc_engine.utils.config_loader import settings
from btc_engine.utils.logging_config import logger


def initialize_database(db_path: str = None) -> None:
    """Initialize database with all required tables
    
    Args:
        db_path: Path to database file. If None, uses settings.database_path
    """
    db_path = db_path or settings.database_path
    logger.info("Initializing database...")
    
    try:
        create_tables(db_path)
        
        # Verify tables
        table_info = get_table_info(db_path)
        logger.info(f"Database initialized with {len(table_info)} tables")
        
        for table_name, row_count in table_info.items():
            logger.debug(f"  {table_name}: {row_count} rows")
            
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def reset_database(db_path: str = None) -> None:
    """Reset database by recreating all tables (WARNING: deletes all data)
    
    Args:
        db_path: Path to database file. If None, uses settings.database_path
    """
    from pathlib import Path
    
    db_path = db_path or settings.database_path
    logger.warning(f"Resetting database at {db_path} - all data will be lost!")
    
    # Delete existing database file
    db_file = Path(db_path)
    if db_file.exists():
        db_file.unlink()
        logger.info("Existing database file deleted")
    
    # Recreate tables
    initialize_database(db_path)
