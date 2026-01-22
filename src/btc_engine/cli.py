"""CLI application using Typer"""

import typer
from typing import Optional
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table

from btc_engine.database.migrations import initialize_database, reset_database
from btc_engine.database.client import db_client
from btc_engine.database.schema import get_table_info
from btc_engine.ingestion import run_deribit_ingestion, run_glassnode_ingestion
from btc_engine.features import (
    calculate_and_store_surface_factors,
    calculate_and_store_risk_neutral,
    calculate_and_store_hedging_pressure,
    calculate_and_store_onchain_indices,
    calculate_and_store_divergence
)
from btc_engine.utils.config_loader import settings
from btc_engine.utils.logging_config import logger

app = typer.Typer(help="BTC Options + Onchain Inference Engine")
console = Console()


@app.command()
def init_db(reset: bool = typer.Option(False, help="Reset database (WARNING: deletes all data)")):
    """Initialize or reset the database"""
    if reset:
        if typer.confirm("Are you sure you want to reset the database? All data will be lost."):
            reset_database()
            console.print("[bold green]Database reset successfully[/bold green]")
        else:
            console.print("[yellow]Reset cancelled[/yellow]")
    else:
        initialize_database()
        console.print("[bold green]Database initialized successfully[/bold green]")
    
    # Show table info
    table_info = get_table_info(settings.database_path)
    
    table = Table(title="Database Tables")
    table.add_column("Table", style="cyan")
    table.add_column("Rows", style="magenta")
    
    for table_name, row_count in table_info.items():
        table.add_row(table_name, str(row_count))
    
    console.print(table)


@app.command()
def ingest_deribit(
    full: bool = typer.Option(True, help="Full snapshot (instruments + tickers) or tickers only")
):
    """Ingest Deribit options data"""
    console.print(f"[bold]Ingesting Deribit data (full={full})...[/bold]")
    
    result = run_deribit_ingestion(full_snapshot=full)
    
    console.print(f"[green]✓ Deribit ingestion complete:[/green]")
    for key, value in result.items():
        if key != "timestamp":
            console.print(f"  {key}: {value}")


@app.command()
def ingest_glassnode(
    days: int = typer.Option(90, help="Number of days to fetch"),
    incremental: bool = typer.Option(True, help="Incremental mode")
):
    """Ingest Glassnode onchain data"""
    console.print(f"[bold]Ingesting Glassnode data ({days} days, incremental={incremental})...[/bold]")
    
    until = int(datetime.now().timestamp())
    since = int((datetime.now() - timedelta(days=days)).timestamp()) if not incremental else None
    
    result = run_glassnode_ingestion(since=since, until=until, incremental=incremental)
    
    console.print(f"[green]✓ Glassnode ingestion complete: {result['records']} records[/green]")


@app.command()
def backfill_history(
    days: int = typer.Option(180, help="Number of days to backfill")
):
    """Generate synthetic historical data for immediate model training"""
    from btc_engine.ingestion.backfill import backfill_deribit_history, backfill_glassnode_history
    
    console.print(f"[bold]Backfilling {days} days of historical data...[/bold]")
    console.print("[dim]This generates synthetic but realistic data using current snapshots as templates[/dim]")
    
    try:
        # Backfill Deribit options data
        console.print("  Generating historical options data...")
        deribit_result = backfill_deribit_history(days=days)
        
        console.print(f"[green]✓ Deribit backfill complete:[/green]")
        console.print(f"  Days: {deribit_result['days_backfilled']}")
        console.print(f"  Instruments: {deribit_result['instruments_created']:,}")
        console.print(f"  Tickers: {deribit_result['tickers_created']:,}")
        console.print(f"  Vol range: {deribit_result['vol_range'][0]:.1%} - {deribit_result['vol_range'][1]:.1%}")
        
        # Ensure Glassnode coverage
        console.print("  Ensuring onchain data coverage...")
        glassnode_result = backfill_glassnode_history(days=days)
        
        console.print(f"[green]✓ Glassnode coverage: {glassnode_result['status']}[/green]")
        
        console.print("\n[bold green]✓ Historical backfill complete![/bold green]")
        console.print("[dim]You can now run: btc-engine build-features && btc-engine train-model[/dim]")
        
    except Exception as e:
        console.print(f"[red]✗ Backfill failed: {e}[/red]")
        logger.exception("Backfill error")
        raise typer.Exit(1)


@app.command()
def build_features(
    timestamp: Optional[str] = typer.Option(None, help="Timestamp (YYYY-MM-DD HH:MM:SS) or 'latest'"),
    all: bool = typer.Option(False, "--all", help="Build features for all historical timestamps")
):
    """Build all features for a timestamp or all historical data"""
    
    if all:
        # Build features for all historical timestamps
        console.print("[bold]Building features for all historical data...[/bold]")
        
        # Get all unique timestamps from raw data
        timestamps_query = """
            SELECT DISTINCT timestamp 
            FROM raw_deribit_ticker_snapshots 
            ORDER BY timestamp
        """
        timestamps_df = db_client.query_to_dataframe(timestamps_query)
        
        if timestamps_df.empty:
            console.print("[yellow]No historical data found[/yellow]")
            return
        
        total = len(timestamps_df)
        console.print(f"[dim]Processing {total} timestamps...[/dim]")
        
        for idx, row in timestamps_df.iterrows():
            ts = row['timestamp']
            
            try:
                # Build features for this timestamp
                calculate_and_store_surface_factors(ts)
                calculate_and_store_risk_neutral(ts)
                calculate_and_store_hedging_pressure(ts)
                calculate_and_store_onchain_indices(ts, ts)
                calculate_and_store_divergence(ts)
                
                # Progress indicator every 10 timestamps
                if (idx + 1) % 10 == 0 or (idx + 1) == total:
                    console.print(f"[dim]  Progress: {idx + 1}/{total} ({100*(idx+1)/total:.1f}%)[/dim]")
                    
            except Exception as e:
                logger.warning(f"Failed to build features for {ts}: {e}")
                continue
        
        console.print(f"[green]✓ Feature building complete: {total} timestamps processed[/green]")
        return
    
    # Single timestamp mode (original behavior)
    if timestamp is None or timestamp == "latest":
        ts = None
        console.print("[bold]Building features for latest data...[/bold]")
    else:
        ts = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        console.print(f"[bold]Building features for {ts}...[/bold]")
    
    try:
        # Options features
        console.print("  Calculating options surface factors...")
        calculate_and_store_surface_factors(ts)
        
        console.print("  Calculating risk-neutral metrics...")
        calculate_and_store_risk_neutral(ts)
        
        console.print("  Calculating hedging pressure...")
        calculate_and_store_hedging_pressure(ts)
        
        # Onchain features (use date range for proper calculation)
        if ts:
            since = ts - timedelta(days=120)
            until = ts
        else:
            since = None
            until = None
        
        console.print("  Calculating onchain indices...")
        calculate_and_store_onchain_indices(since, until)
        
        console.print("  Detecting divergence...")
        if ts is None:
            latest_ts = db_client.get_latest_timestamp("features_options_surface")
            if latest_ts:
                ts = latest_ts
        
        if ts:
            calculate_and_store_divergence(ts)
        
        console.print("[green]✓ Feature building complete[/green]")
        
    except Exception as e:
        console.print(f"[red]✗ Feature building failed: {e}[/red]")
        logger.error(f"Feature building error: {e}", exc_info=True)
        raise typer.Exit(code=1)



@app.command()
def train_model(
    days: int = typer.Option(180, help="Days of training data"),
    version: Optional[str] = typer.Option(None, help="Model version")
):
    """Train state-space model for regime inference"""
    console.print(f"[bold]Training state-space model with {days} days of data...[/bold]")
    
    try:
        from btc_engine.models import train_and_save_model
        
        until = datetime.now()
        since = until - timedelta(days=days)
        
        result = train_and_save_model(since=since, until=until, model_version=version)
        
        console.print(f"[green]✓ Model training complete[/green]")
        console.print(f"  Model version: {result['model_version']}")
        console.print(f"  Observations: {result['n_observations']}")
        
    except Exception as e:
        console.print(f"[red]✗ Model training failed: {e}[/red]")
        logger.error(f"Model training error: {e}", exc_info=True)
        raise typer.Exit(code=1)


@app.command()
def forecast(
    days: int = typer.Option(180, help="Days of training data"),
    version: Optional[str] = typer.Option(None, help="Model version")
):
    """Generate distributional forecasts"""
    console.print(f"[bold]Generating forecasts...[/bold]")
    
    try:
        from btc_engine.models import train_and_forecast
        
        until = datetime.now()
        since = until - timedelta(days=days)
        
        result = train_and_forecast(since=since, until=until, model_version=version)
        
        console.print(f"[green]✓ Forecasting complete[/green]")
        console.print(f"  Model version: {result['model_version']}")
        console.print(f"  Horizons: {result['n_horizons']}")
        
    except Exception as e:
        console.print(f"[red]✗ Forecasting failed: {e}[/red]")
        logger.error(f"Forecasting error: {e}", exc_info=True)
        raise typer.Exit(code=1)


@app.command()
def evaluate(
    horizons: Optional[str] = typer.Option("24h,7d", help="Comma-separated horizons"),
    days: int = typer.Option(90, help="Days of backtest window")
):
    """Run model evaluation and backtesting"""
    console.print(f"[bold]Running evaluation...[/bold]")
    
    try:
        from btc_engine.models.evaluation import run_model_evaluation
        
        horizon_list = horizons.split(',') if horizons else None
        summary = run_model_evaluation(horizons=horizon_list, lookback_days=days)
        
        console.print(f"[green]✓ Evaluation complete[/green]")
        for horizon, metrics in summary.items():
            console.print(f"{horizon}: {len(metrics)} metrics calculated")
        
    except Exception as e:
        console.print(f"[red]✗ Evaluation failed: {e}[/red]")
        logger.error(f"Evaluation error: {e}", exc_info=True)
        raise typer.Exit(code=1)


@app.command()
def newsletter(
    week: Optional[str] = typer.Option(None, help="Week date (YYYY-MM-DD)")
):
    """Generate weekly risk report newsletter"""
    console.print(f"[bold]Generating newsletter...[/bold]")
    
    try:
        from btc_engine.newsletter import generate_weekly_newsletter
        
        week_date = datetime.strptime(week, "%Y-%m-%d") if week else None
        output_path = generate_weekly_newsletter(week_date=week_date)
        
        console.print(f"[green]✓ Newsletter generated[/green]")
        console.print(f"  Saved to: {output_path}")
        
    except Exception as e:
        console.print(f"[red]✗ Newsletter generation failed: {e}[/red]")
        logger.error(f"Newsletter error: {e}", exc_info=True)
        raise typer.Exit(code=1)


@app.command()
def dashboard(
    port: int = typer.Option(8501, help="Dashboard port"),
    host: str = typer.Option("0.0.0.0", help="Dashboard host")
):
    """Launch Streamlit dashboard"""
    import subprocess
    
    console.print(f"[bold]Launching dashboard at http://{host}:{port}[/bold]")
    
    dashboard_path = "src/btc_engine/dashboard/app.py"
    subprocess.run([
        "streamlit", "run", dashboard_path,
        "--server.port", str(port),
        "--server.address", host
    ])


@app.command()
def demo():
    """Run demo with sample data"""
    console.print("[bold yellow]Demo mode not yet implemented[/bold yellow]")
    console.print("Please use real API keys for now.")


@app.command()
def status():
    """Show system status"""
    table_info = get_table_info(settings.database_path)
    
    table = Table(title="System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="white")
    
    # Database
    total_rows = sum(table_info.values())
    table.add_row(
        "Database",
        "✓ Connected",
        f"{len(table_info)} tables, {total_rows} total rows"
    )
    
    # API Keys
    api_status = []
    if settings.deribit_api_key:
        api_status.append("Deribit")
    if settings.glassnode_api_key:
        api_status.append("Glassnode")
    
    table.add_row(
        "API Keys",
        "✓ Configured" if api_status else "✗ Missing",
        ", ".join(api_status) if api_status else "No keys configured"
    )
    
    # Data freshness
    latest_deribit = db_client.get_latest_timestamp("raw_deribit_ticker_snapshots")
    latest_glassnode = db_client.get_latest_timestamp("raw_glassnode_metrics")
    
    if latest_deribit:
        age = datetime.now() - latest_deribit
        table.add_row(
            "Deribit Data",
            "✓ Available",
            f"Latest: {latest_deribit.strftime('%Y-%m-%d %H:%M')} ({age.total_seconds()/3600:.1f}h ago)"
        )
    else:
        table.add_row("Deribit Data", "✗ No data", "Run: btc-engine ingest-deribit")
    
    if latest_glassnode:
        age = datetime.now() - latest_glassnode
        table.add_row(
            "Glassnode Data",
            "✓ Available",
            f"Latest: {latest_glassnode.strftime('%Y-%m-%d')} ({age.days}d ago)"
        )
    else:
        table.add_row("Glassnode Data", "✗ No data", "Run: btc-engine ingest-glassnode")
    
    console.print(table)


if __name__ == "__main__":
    app()
