# GitHub Actions Setup

This project uses GitHub Actions to automatically update data and retrain models.

## Workflows

### Daily Data Update (`daily-update.yml`)
- **Schedule**: Runs daily at 12:00 UTC
- **Actions**:
  1. Ingest latest Deribit options data
  2. Ingest Glassnode onchain data (last 7 days)
  3. Build features from new data
  4. Save database as artifact

### Weekly Model Retraining
- **Schedule**: Runs every Sunday after data update
- **Actions**:
  1. Train state-space model on 180 days of data
  2. Generate forecasts
  3. Save model artifacts

## Setup Instructions

### 1. Add API Keys to GitHub Secrets

Go to your repository **Settings → Secrets and variables → Actions** and add:

- `DERIBIT_API_KEY`: Your Deribit API key
- `DERIBIT_API_SECRET`: Your Deribit API secret
- `GLASSNODE_API_KEY`: Your Glassnode API key

### 2. Enable GitHub Actions

1. Go to your repository **Actions** tab
2. Enable workflows if prompted
3. The workflow will run automatically on schedule

### 3. Manual Trigger (Optional)

You can manually trigger the workflow:
1. Go to **Actions** tab
2. Select "Daily Data Update"
3. Click "Run workflow"

## Artifacts

The workflow saves artifacts after each run:
- **Database**: `btc-engine-data-{run_number}` (7 days retention)
- **Models**: `btc-engine-model-{run_number}` (30 days retention)

Download these from the Actions run page to use locally.

## Cost Considerations

- **GitHub Actions**: 2,000 free minutes/month for public repos
- **API Calls**: 
  - Deribit: Free public API (rate limited)
  - Glassnode: Depends on your plan
- **Estimated usage**: ~5 minutes/day = 150 minutes/month (well within free tier)

## Monitoring

Check the **Actions** tab regularly to ensure workflows are running successfully. You'll receive email notifications on failures.
