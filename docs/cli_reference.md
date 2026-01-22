# CLI Quick Reference

## Available Commands

```bash
# Database
btc-engine init-db              # Initialize database
btc-engine init-db --reset      # Reset database (WARNING: deletes all data)

# Data Ingestion
btc-engine ingest-deribit       # Fetch Deribit options data
btc-engine ingest-glassnode     # Fetch Glassnode onchain data
btc-engine ingest-glassnode --days 30  # Fetch last 30 days

# Feature Engineering
btc-engine build-features       # Build all features for latest data
btc-engine build-features --timestamp "2024-01-15 12:00:00"

# Modeling (Python API recommended for now)
# btc-engine train-model          # Train state-space model
# btc-engine forecast             # Generate forecasts
# btc-engine evaluate             # Run backtesting
# btc-engine newsletter           # Generate weekly report

# Visualization
btc-engine dashboard            # Launch dashboard at localhost:8501
btc-engine dashboard --port 8080

# System
btc-engine status               # Show system status
```

## Python API Usage

For model training, forecasting, and evaluation, you can use the Python API directly:

```python
from btc_engine.models import train_and_save_model, train_and_forecast
from btc_engine.models.evaluation import run_model_evaluation
from btc_engine.newsletter import generate_weekly_newsletter
from datetime import datetime, timedelta

# Train model
until = datetime.now()
since = until - timedelta(days=180)
result = train_and_save_model(since, until)

# Generate forecasts
forecast_result = train_and_forecast(since, until)

# Run evaluation
eval_summary = run_model_evaluation(lookback_days=90)

# Generate newsletter
newsletter_path = generate_weekly_newsletter()
```

## Common Workflows

### Initial Setup
```bash
btc-engine init-db
btc-engine ing est-deribit
btc-engine ingest-glassnode --days 90
btc-engine build-features
```

### Daily Update
```bash
btc-engine ingest-deribit
btc-engine ingest-glassnode --incremental
btc-engine build-features
```

### View Results
```bash
btc-engine dashboard
# Open browser to http://localhost:8501
```
