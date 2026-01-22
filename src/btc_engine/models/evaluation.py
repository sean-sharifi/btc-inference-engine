"""Model evaluation metrics and backtesting framework"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.stats import norm

from btc_engine.database.client import db_client
from btc_engine.utils.logging_config import logger


def calculate_crps(
    forecasts: np.ndarray,
    actuals: np.ndarray,
    quantiles: List[float]
) -> float:
    """Calculate Continuous Ranked Probability Score
    
    Args:
        forecasts: Array of quantile forecasts (n_samples × n_quantiles)
        actuals: Array of actual values
        quantiles: List of quantile levels
        
    Returns:
        CRPS score (lower is better)
    """
    n = len(actuals)
    crps_sum = 0.0
    
    for i in range(n):
        actual = actuals[i]
        forecast_quantiles = forecasts[i]
        
        # CRPS via quantile integration
        crps_i = 0.0
        for j, q in enumerate(quantiles):
            forecast_q = forecast_quantiles[j]
            indicator = 1.0 if actual <= forecast_q else 0.0
            crps_i += (indicator - q) * (forecast_q - actual)
        
        crps_sum += abs(crps_i)
    
    return crps_sum / n


def calculate_pinball_loss(
    forecast: float,
    actual: float,
    quantile: float
) -> float:
    """Calculate pinball loss for single quantile forecast
    
    Args:
        forecast: Forecasted quantile value
        actual: Actual value
        quantile: Quantile level (0-1)
        
    Returns:
        Pinball loss
    """
    error = actual - forecast
    if error >= 0:
        return quantile * error
    else:
        return (quantile - 1) * error


def calculate_hit_rate(
    forecasts: np.ndarray,
    actuals: np.ndarray,
    quantile: float,
    tolerance: float = 0.02
) -> float:
    """Calculate hit rate for quantile forecast
    
    Args:
        forecasts: Array of quantile forecasts
        actuals: Array of actual values
        quantile: Target quantile
        tolerance: Acceptable tolerance
        
    Returns:
        Hit rate (should be close to quantile level)
    """
    hits = np.sum(actuals <= forecasts)
    hit_rate = hits / len(actuals)
    
    return hit_rate


def calculate_calibration_curve(
    forecasts: np.ndarray,
    actuals: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate calibration curve for probabilistic forecasts
    
    Args:
        forecasts: Forecasted probabilities
        actuals: Binary outcomes
        n_bins: Number of bins
        
    Returns:
        Tuple of (predicted_probs, observed_freqs)
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    predicted_probs = []
    observed_freqs = []
    
    for i in range(n_bins):
        mask = (forecasts >= bins[i]) & (forecasts < bins[i+1])
        if np.sum(mask) > 0:
            predicted_probs.append(np.mean(forecasts[mask]))
            observed_freqs.append(np.mean(actuals[mask]))
    
    return np.array(predicted_probs), np.array(observed_freqs)


class ModelEvaluator:
    """Evaluate forecasting model performance"""
    
    def __init__(self):
        """Initialize evaluator"""
        from btc_engine.utils.config_loader import load_yaml_config
        config = load_yaml_config("model")
        self.config = config["evaluation"]
        
        self.metrics = self.config["metrics"]
        self.n_bins = self.config["calibration"]["n_bins"]
    
    def fetch_forecasts_and_actuals(
        self,
        horizon: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Fetch forecasts and actual outcomes
        
        Args:
            horizon: Forecast horizon
            since: Start date
            until: End date
            
        Returns:
            DataFrame with forecasts and actuals
        """
        query = """
            SELECT 
                f.forecast_timestamp,
                f.target_timestamp,
                f.horizon,
                f.quantile_05,
                f.quantile_25,
                f.quantile_50,
                f.quantile_75,
                f.quantile_95
            FROM forecasts f
            WHERE f.horizon = ?
            AND f.forecast_timestamp BETWEEN ? AND ?
            ORDER BY f.forecast_timestamp
        """
        
        if since is None:
            since = datetime.now() - timedelta(days=90)
        if until is None:
            until = datetime.now()
        
        df = db_client.query_to_dataframe(query, (horizon, since, until))
        
        if len(df) == 0:
            logger.warning(f"No forecasts found for horizon {horizon}")
            return pd.DataFrame()
        
        # Fetch actual returns for target timestamps
        actuals = []
        for _, row in df.iterrows():
            target_ts = row['target_timestamp']
            forecast_ts = row['forecast_timestamp']
            
            # Get price at forecast time and target time
            price_query = """
                SELECT AVG(underlying_price) as price
                FROM raw_deribit_ticker_snapshots
                WHERE timestamp = ?
            """
            
            price_forecast = db_client.query_to_dataframe(price_query, (forecast_ts,))
            price_target = db_client.query_to_dataframe(price_query, (target_ts,))
            
            if len(price_forecast) > 0 and len(price_target) > 0:
                p0 = price_forecast['price'].iloc[0]
                p1 = price_target['price'].iloc[0]
                actual_return = (p1 - p0) / p0
            else:
                actual_return = np.nan
            
            actuals.append(actual_return)
        
        df['actual_return'] = actuals
        df = df.dropna(subset=['actual_return'])
        
        return df
    
    def evaluate_horizon(
        self,
        horizon: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Evaluate forecasts for given horizon
        
        Args:
            horizon: Forecast horizon
            since: Start date
            until: End date
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating forecasts for horizon {horizon}")
        
        df = self.fetch_forecasts_and_actuals(horizon, since, until)
        
        if len(df) < 10:
            logger.warning(f"Insufficient data for evaluation ({len(df)} samples)")
            return {}
        
        results = {}
        
        # Extract quantile forecasts and actuals
        quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
        forecast_matrix = df[['quantile_05', 'quantile_25', 'quantile_50', 'quantile_75', 'quantile_95']].values
        actuals = df['actual_return'].values
        
        # CRPS
        if 'crps' in self.metrics:
            crps = calculate_crps(forecast_matrix, actuals, quantiles)
            results['crps'] = crps
        
        # Pinball loss per quantile
        if 'pinball_loss' in self.metrics:
            for i, q in enumerate(quantiles):
                forecasts_q = forecast_matrix[:, i]
                pinball = np.mean([
                    calculate_pinball_loss(f, a, q)
                    for f, a in zip(forecasts_q, actuals)
                ])
                results[f'pinball_loss_q{int(q*100)}'] = pinball
        
        # Hit rates
        if 'hit_rate_5pct' in self.metrics:
            hit_rate_05 = calculate_hit_rate(forecast_matrix[:, 0], actuals, 0.05)
            results['hit_rate_5pct'] = hit_rate_05
        
        if 'hit_rate_95pct' in self.metrics:
            hit_rate_95 = calculate_hit_rate(forecast_matrix[:, 4], actuals, 0.95)
            results['hit_rate_95pct'] = hit_rate_95
        
        # MAE and RMSE on median
        if 'mae' in self.metrics:
            mae = np.mean(np.abs(forecast_matrix[:, 2] - actuals))
            results['mae'] = mae
        
        if 'rmse' in self.metrics:
            rmse = np.sqrt(np.mean((forecast_matrix[:, 2] - actuals)**2))
            results['rmse'] = rmse
        
        logger.info(f"Evaluation complete for {horizon}: {results}")
        
        return results
    
    def run_backtest(
        self,
        horizons: Optional[List[str]] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        model_version: str = None
    ) -> pd.DataFrame:
        """Run full backtest evaluation
        
        Args:
            horizons: List of horizons to evaluate
            since: Start date
            until: End date
            model_version: Model version to evaluate
            
        Returns:
            DataFrame with all evaluation results
        """
        if horizons is None:
            horizons = ['24h', '7d']
        
        if model_version is None:
            model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Running backtest for model {model_version}")
        
        results = []
        
        for horizon in horizons:
            metrics = self.evaluate_horizon(horizon, since, until)
            
            for metric_name, metric_value in metrics.items():
                record = {
                    'evaluation_timestamp': datetime.now(),
                    'backtest_window_start': since,
                    'backtest_window_end': until,
                    'metric_name': metric_name,
                    'metric_value': metric_value,
                    'horizon': horizon,
                    'model_version': model_version
                }
                results.append(record)
        
        df_results = pd.DataFrame(results)
        
        # Save to database
        if len(df_results) > 0:
            db_client.insert_dataframe("evaluations", df_results, if_exists="append")
            logger.info(f"Saved {len(df_results)} evaluation metrics")
        
        return df_results


def run_model_evaluation(
    horizons: Optional[List[str]] = None,
    lookback_days: int = 90
) -> Dict:
    """Run model evaluation and return summary
    
    Args:
        horizons: List of horizons to evaluate
        lookback_days: Days of history to evaluate
        
    Returns:
        Evaluation summary
    """
    evaluator = ModelEvaluator()
    
    until = datetime.now()
    since = until - timedelta(days=lookback_days)
    
    results_df = evaluator.run_backtest(horizons, since, until)
    
    # Create summary
    summary = {}
    for horizon in (horizons or ['24h', '7d']):
        horizon_metrics = results_df[results_df['horizon'] == horizon]
        summary[horizon] = {
            row['metric_name']: row['metric_value']
            for _, row in horizon_metrics.iterrows()
        }
    
    return summary
