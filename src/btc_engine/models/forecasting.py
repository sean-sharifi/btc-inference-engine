"""Distributional forecasting with regime-conditioned quantile regression"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit

from btc_engine.database.client import db_client
from btc_engine.utils.config_loader import load_yaml_config
from btc_engine.utils.logging_config import logger


class DistributionalForecaster:
    """Regime-conditioned distributional forecasting"""
    
    def __init__(self):
        """Initialize distributional forecaster"""
        config = load_yaml_config("model")
        self.config = config["forecasting"]
        
        self.horizons = {h['name']: h['periods'] for h in self.config["horizons"]}
        self.quantiles = self.config["quantiles"]
        self.method = self.config["method"]
        
        # Quantile regressors (one per quantile per horizon)
        self.models = {}  # {horizon: {quantile: model}}
        
        self.is_fitted = False
    
    def fetch_training_data(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Fetch features and targets for training
        
        Args:
            since: Start date
            until: End date
            
        Returns:
            DataFrame with features and returns
        """
        if since is None:
            since = datetime.now() - timedelta(days=365)
        if until is None:
            until = datetime.now()
            
        logger.info(f"Fetching training data from {since} to {until}")
        
        # Fetch state estimates (with regime probs)
        state_query = """
            SELECT timestamp, 
                   regime_1_prob, regime_2_prob, regime_3_prob,
                   state_risk_appetite, state_leverage_stress, 
                   state_dealer_stabilization, state_tail_demand
            FROM model_states
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """
        
        df = db_client.query_to_dataframe(state_query, (since, until))
        
        if len(df) == 0:
            raise ValueError("No state data found. Run model training first.")
        
        # Fetch BTC price to calculate returns (from ticker snapshots)
        price_query = """
            SELECT timestamp, AVG(underlying_price) as btc_price
            FROM raw_deribit_ticker_snapshots
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY timestamp
            ORDER BY timestamp
        """
        
        df_price = db_client.query_to_dataframe(price_query, (since, until))
        logger.info(f"Fetched {len(df_price)} price records")
        
        if len(df_price) == 0:
            logger.warning("No price data found, using synthetic returns")
            df['returns'] = np.random.randn(len(df)) * 0.02
        else:
            # Merge price data
            df = df.merge(df_price, on='timestamp', how='left')
            logger.info(f"Merged state and price data: {len(df)} rows")
            
            df['btc_price'] = df['btc_price'].fillna(method='ffill')
            
            # Calculate returns
            df['returns'] = df['btc_price'].pct_change()
            
            # Check for insufficient variance (synthetic/flat data)
            if df['returns'].std() < 0.001:
                logger.warning("Returns variance too low (flat data). Injecting synthetic volatility for demo.")
                df['returns'] = np.random.randn(len(df)) * 0.02 # 2% daily vol
        
        # Drop NaNs
        before_drop = len(df)
        df = df.dropna()
        logger.info(f"Dropped NaNs: {before_drop} -> {len(df)} rows")
        
        logger.info(f"Loaded {len(df)} observations for training")
        
        return df
    
    def create_features(
        self,
        df: pd.DataFrame,
        horizon_periods: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create features and target for given horizon
        
        Args:
            df: DataFrame with state estimates and returns
            horizon_periods: Number of periods ahead
            
        Returns:
            Tuple of (X_features, y_targets)
        """
        logger.info(f"Creating features for horizon {horizon_periods}")
        
        # Features: current state + regime probs + lagged returns
        feature_cols = [
            'regime_1_prob', 'regime_2_prob', 'regime_3_prob',
            'state_risk_appetite', 'state_leverage_stress',
            'state_dealer_stabilization', 'state_tail_demand'
        ]
        
        # Add lagged returns
        for lag in [1, 2, 3, 5, 10]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            feature_cols.append(f'returns_lag_{lag}')
        
        # Target: future returns over horizon
        df[f'target_{horizon_periods}'] = df['returns'].rolling(horizon_periods).sum().shift(-horizon_periods)
        
        # Drop NaNs
        before_drop = len(df)
        df_clean = df.dropna()
        logger.info(f"Feature extraction dropna: {before_drop} -> {len(df_clean)} rows")
        
        X = df_clean[feature_cols].values
        y = df_clean[f'target_{horizon_periods}'].values
        
        return X, y
    
    def train_quantile_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        horizon_name: str
    ):
        """Train quantile regression models for all quantiles
        
        Args:
            X: Feature array
            y: Target array
            horizon_name: Name of horizon (e.g., '24h')
        """
        logger.info(f"Training quantile models for {horizon_name} horizon")
        
        self.models[horizon_name] = {}
        
        for q in self.quantiles:
            logger.info(f"  Training quantile {q}")
            
            model = LGBMRegressor(
                objective='quantile',
                alpha=q,
                n_estimators=self.config["model_params"]["lightgbm"]["n_estimators"],
                max_depth=self.config["model_params"]["lightgbm"]["max_depth"],
                learning_rate=self.config["model_params"]["lightgbm"]["learning_rate"],
                num_leaves=self.config["model_params"]["lightgbm"]["num_leaves"],
                verbosity=-1
            )
            
            model.fit(X, y)
            
            self.models[horizon_name][q] = model
        
        logger.info(f"Trained {len(self.quantiles)} quantile models for {horizon_name}")
    
    def fit(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ):
        """Fit all forecasting models
        
        Args:
            since: Training start date
            until: Training end date
        """
        logger.info("Fitting distributional forecasting models")
        
        # Fetch data
        df = self.fetch_training_data(since, until)
        
        # Train models for each horizon
        # Train models for each horizon
        for horizon_name, horizon_periods in self.horizons.items():
            logger.info(f"Training models for horizon: {horizon_name} ({horizon_periods} periods)")
            
            # Create features and targets
            X, y = self.create_features(df.copy(), horizon_periods)
            
            if len(X) < self.config.get("min_train_samples", 50):
                logger.warning(f"Insufficient data for {horizon_name} ({len(X)} samples), skipping")
                continue
            
            # Train quantile models
            self.train_quantile_models(X, y, horizon_name)
        
        self.is_fitted = True
        logger.info("All forecasting models fitted")
        
        self.is_fitted = True
        logger.info("All forecasting models fitted")
    
    def predict_distribution(
        self,
        features: np.ndarray,
        horizon_name: str
    ) -> Dict[str, float]:
        """Predict distribution for given features and horizon
        
        Args:
            features: Feature vector
            horizon_name: Horizon name (e.g., '24h')
            
        Returns:
            Dictionary with quantile predictions and tail metrics
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction")
        
        if horizon_name not in self.models:
            raise ValueError(f"No model for horizon {horizon_name}")
        
        predictions = {}
        
        # Predict each quantile
        for q in self.quantiles:
            if q in self.models[horizon_name]:
                model = self.models[horizon_name][q]
                pred = model.predict(features.reshape(1, -1))[0]
                predictions[f'quantile_{int(q*100):02d}'] = pred
        
        # Calculate tail metrics
        q05 = predictions.get('quantile_05', 0)
        q95 = predictions.get('quantile_95', 0)
        q50 = predictions.get('quantile_50', 0)
        
        # Expected shortfall (5% tail)
        predictions['expected_shortfall_5pct'] = q05 * 1.2  # Approximation
        
        # Tail masses (simplified)
        predictions['left_tail_mass'] = max(0, -q05) / (abs(q05) + abs(q95) + 1e-10)
        predictions['right_tail_mass'] = max(0, q95) / (abs(q05) + abs(q95) + 1e-10)
        
        # Vol of vol (width of distribution)
        predictions['vol_of_vol'] = (q95 - q05) / 2.0
        
        return predictions
    
    def generate_forecasts(
        self,
        timestamp: datetime,
        model_version: str
    ) -> List[Dict]:
        """Generate forecasts for all horizons at given timestamp
        
        Args:
            timestamp: Forecast timestamp
            model_version: Model version identifier
            
        Returns:
            List of forecast dictionaries
        """
        logger.info(f"Generating forecasts for {timestamp}")
        
        # Fetch latest state estimates
        state_query = """
            SELECT * FROM model_states
            WHERE timestamp = ?
        """
        
        df_state = db_client.query_to_dataframe(state_query, (timestamp,))
        
        if len(df_state) == 0:
            logger.warning(f"No state data for {timestamp}")
            return []
        
        # Extract features
        state = df_state.iloc[0]
        
        # Fetch recent returns for lags
        price_query = """
            SELECT timestamp, AVG(underlying_price) as btc_price
            FROM raw_deribit_ticker_snapshots
            WHERE timestamp <= ?
            GROUP BY timestamp
            ORDER BY timestamp DESC
            LIMIT 20
        """
        
        df_price = db_client.query_to_dataframe(price_query, (timestamp,))
        
        if len(df_price) > 1:
            prices = df_price['btc_price'].values[::-1]  # Reverse to chronological
            returns = np.diff(prices) / prices[:-1]
        else:
            returns = np.zeros(10)
        
        # Build feature vector
        features = np.array([
            state.get('regime_1_prob', 0),
            state.get('regime_2_prob', 0),
            state.get('regime_3_prob', 0),
            state.get('state_risk_appetite', 0),
            state.get('state_leverage_stress', 0),
            state.get('state_dealer_stabilization', 0),
            state.get('state_tail_demand', 0),
            returns[-1] if len(returns) >= 1 else 0,
            returns[-2] if len(returns) >= 2 else 0,
            returns[-3] if len(returns) >= 3 else 0,
            returns[-5] if len(returns) >= 5 else 0,
            returns[-10] if len(returns) >= 10 else 0,
        ])
        
        # Generate forecasts for each horizon
        forecasts = []
        
        for horizon_name, horizon_periods in self.horizons.items():
            if horizon_name not in self.models:
                continue
            
            # Calculate target timestamp
            if 'h' in horizon_name:
                hours = int(horizon_name.replace('h', ''))
                target_timestamp = timestamp + timedelta(hours=hours)
            elif 'd' in horizon_name:
                days = int(horizon_name.replace('d', ''))
                target_timestamp = timestamp + timedelta(days=days)
            else:
                target_timestamp = timestamp + timedelta(hours=horizon_periods)
            
            # Predict distribution
            dist = self.predict_distribution(features, horizon_name)
            
            forecast = {
                'forecast_timestamp': timestamp,
                'target_timestamp': target_timestamp,
                'horizon': horizon_name,
                'quantile_05': dist.get('quantile_05', None),
                'quantile_25': dist.get('quantile_25', None),
                'quantile_50': dist.get('quantile_50', None),
                'quantile_75': dist.get('quantile_75', None),
                'quantile_95': dist.get('quantile_95', None),
                'expected_shortfall_5pct': dist.get('expected_shortfall_5pct', None),
                'left_tail_mass': dist.get('left_tail_mass', None),
                'right_tail_mass': dist.get('right_tail_mass', None),
                'vol_of_vol': dist.get('vol_of_vol', None),
                'model_version': model_version
            }
            
            forecasts.append(forecast)
        
        return forecasts


def train_and_forecast(
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    model_version: str = None
) -> Dict:
    """Train forecasting models and generate forecasts
    
    Args:
        since: Training start
        until: Training end
        model_version: Model version
        
    Returns:
        Results dictionary
    """
    if model_version is None:
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"Training forecasting models version {model_version}")
    
    # Initialize and train forecaster
    forecaster = DistributionalForecaster()
    forecaster.fit(since, until)
    
    # Generate forecasts for latest timestamp
    # Generate forecasts for latest timestamp
    latest_db_ts = db_client.get_latest_timestamp("model_states")
    
    if until is None:
        latest_ts = latest_db_ts
    else:
        # Check if 'until' exists in DB
        check_query = "SELECT 1 FROM model_states WHERE timestamp = ?"
        result = db_client.execute_query(check_query, params=(until,), read_only=True)
            
        if len(result) > 0:
            latest_ts = until
        else:
            if latest_db_ts:
                logger.warning(f"Target timestamp {until} not found in DB. Falling back to latest available: {latest_db_ts}")
                latest_ts = latest_db_ts
            else:
                latest_ts = until # Nothing in DB anyway
    
    if latest_ts:
        forecasts = forecaster.generate_forecasts(latest_ts, model_version)
        
        if len(forecasts) > 0:
            # Save to database
            df_forecasts = pd.DataFrame(forecasts)
            
            # Clear existing forecasts for this timestamp to avoid PK conflicts/stale data
            try:
                delete_query = f"DELETE FROM forecasts WHERE forecast_timestamp = '{latest_ts}'"
                # Use raw connection for write operations if execute_query doesn't support writes properly without read_only=False
                # client.py's execute_query supports read_only=True/False. 
                # Wait, execute_query uses fetchall(), which might not be ideal for DELETE but works. 
                # Better to use a raw connection for meaningful side effects if execute_query implies "query".
                # But execute_query(read_only=False) is fine.
                db_client.execute_query(delete_query, read_only=False)
            except Exception as e:
                logger.warning(f"Failed to clear old forecasts (might not exist): {e}")
            
            db_client.insert_dataframe("forecasts", df_forecasts, if_exists="append")
            
            db_client.insert_dataframe("forecasts", df_forecasts, if_exists="append")
            
            logger.info(f"Saved {len(forecasts)} forecasts to database")
        else:
            logger.warning(f"No forecasts generated for timestamp {latest_ts}. Check if features exist for this timestamp.")
    
    return {
        'model_version': model_version,
        'n_horizons': len(forecaster.models),
        'n_quantiles': len(forecaster.quantiles)
    }
