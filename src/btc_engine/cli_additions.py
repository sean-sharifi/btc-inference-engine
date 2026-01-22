# Additional CLI commands to be integrated

@app.command()
def train_model(
    days: int = typer.Option(180, help="Days of training data"),
    version: Optional[str] = typer.Option(None, help="Model version")
):
    """Train state-space model"""
    console.print(f"[bold]Training model with {days} days...[/bold]")
    try:
        from btc_engine.models import train_and_save_model
        until = datetime.now()
        since = until - timedelta(days=days)
        result = train_and_save_model(since, until, version)
        console.print(f"[green]✓ Complete: {result['model_version']}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def forecast(days: int = typer.Option(180)):
    """Generate forecasts"""
    console.print("[bold]Generating forecasts...[/bold]")
    try:
        from btc_engine.models import train_and_forecast
        until = datetime.now()
        since = until - timedelta(days=days)
        result = train_and_forecast(since, until)
        console.print(f"[green]✓ Complete[/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def evaluate(days: int = typer.Option(90)):
    """Run evaluation"""
    console.print("[bold]Running evaluation...[/bold]")
    try:
        from btc_engine.models.evaluation import run_model_evaluation
        summary = run_model_evaluation(lookback_days=days)
        console.print(f"[green]✓ Complete[/green]")
        for horizon, metrics in summary.items():
            console.print(f"\n{horizon}: {metrics}")
    except Exception as e:
        console.print(f"[red]✗ Failed: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def newsletter(week: Optional[str] = typer.Option(None)):
    """Generate newsletter"""
    console.print("[bold]Generating newsletter...[/bold]")
    try:
        from btc_engine.newsletter import generate_weekly_newsletter
        week_date = datetime.strptime(week, "%Y-%m-%d") if week else None
        output_path = generate_weekly_newsletter(week_date)
        console.print(f"[green]✓ Saved to: {output_path}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed: {e}[/red]")
        raise typer.Exit(1)
