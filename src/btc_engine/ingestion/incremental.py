"""Incremental ingestion utilities and checkpoint management"""

from datetime import datetime
from typing import Optional, Dict
import pandas as pd

from btc_engine.database.client import db_client
from btc_engine.utils.logging_config import logger


def update_checkpoint(
    task_name: str,
    status: str = "success",
    records_processed: int = 0,
    error_message: Optional[str] = None
) -> None:
    """Update pipeline checkpoint for a task
    
    Args:
        task_name: Name of the task
        status: Status ('success', 'running', 'failed')
        records_processed: Number of records processed
        error_message: Error message if failed
    """
    now = datetime.now()
    
    # Check if checkpoint exists
    existing = db_client.execute_query(
        "SELECT * FROM pipeline_checkpoints WHERE task_name = ?",
        (task_name,)
    )
    
    if existing:
        # Update existing checkpoint
        if status == "success":
            db_client.execute_query(
                """UPDATE pipeline_checkpoints 
                SET last_run_timestamp = ?, 
                    last_success_timestamp = ?, 
                    status = ?, 
                    records_processed = ?,
                    error_message = ?
                WHERE task_name = ?""",
                (now, now, status, records_processed, error_message, task_name),
                read_only=False
            )
        else:
            db_client.execute_query(
                """UPDATE pipeline_checkpoints 
                SET last_run_timestamp = ?, 
                    status = ?, 
                    records_processed = ?,
                    error_message = ?
                WHERE task_name = ?""",
                (now, status, records_processed, error_message, task_name),
                read_only=False
            )
    else:
        # Insert new checkpoint
        success_ts = now if status == "success" else None
        db_client.execute_query(
            """INSERT INTO pipeline_checkpoints 
            (task_name, last_run_timestamp, last_success_timestamp, status, records_processed, error_message)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (task_name, now, success_ts, status, records_processed, error_message),
            read_only=False
        )
    
    logger.debug(f"Updated checkpoint for {task_name}: {status}")


def get_checkpoint(task_name: str) -> Optional[Dict]:
    """Get checkpoint information for a task
    
    Args:
        task_name: Name of the task
        
    Returns:
        Dictionary with checkpoint info or None if not found
    """
    result = db_client.execute_query(
        "SELECT * FROM pipeline_checkpoints WHERE task_name = ?",
        (task_name,)
    )
    
    if not result:
        return None
    
    row = result[0]
    return {
        "task_name": row[0],
        "last_run_timestamp": row[1],
        "last_success_timestamp": row[2],
        "status": row[3],
        "records_processed": row[4],
        "error_message": row[5]
    }


def should_run_task(task_name: str, min_interval_hours: int = 1) -> bool:
    """Check if a task should run based on checkpoint
    
    Args:
        task_name: Name of the task
        min_interval_hours: Minimum hours between runs
        
    Returns:
        True if task should run
    """
    checkpoint = get_checkpoint(task_name)
    
    if not checkpoint:
        logger.info(f"No checkpoint for {task_name}, should run")
        return True
    
    if checkpoint["status"] == "failed":
        logger.info(f"Last run of {task_name} failed, should retry")
        return True
    
    if checkpoint["last_success_timestamp"]:
        elapsed = datetime.now() - checkpoint["last_success_timestamp"]
        elapsed_hours = elapsed.total_seconds() / 3600
        
        if elapsed_hours < min_interval_hours:
            logger.info(f"{task_name} ran {elapsed_hours:.1f}h ago, skipping (min interval: {min_interval_hours}h)")
            return False
    
    return True
