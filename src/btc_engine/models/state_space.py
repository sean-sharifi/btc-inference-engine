"""State-space model for regime switching and latent state inference"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from btc_engine.database.client import db_client
from btc_engine.utils.config_loader import load_yaml_config
from btc_engine.utils.logging_config import logger


class SwitchingKalmanFilter:
    """Switching Kalman Filter for regime-switching state-space model
    
    Implements a discrete regime-switching model with Kalman filtering
    for continuous states within each regime.
    """
    
    def __init__(self):
        """Initialize Switching Kalman Filter"""
        config = load_yaml_config("model")
        self.config = config["state_space"]
        
        self.n_regimes = self.config["n_regimes"]
        self.regime_names = self.config["regime_names"]
        self.state_dim = self.config["state_dim"]
        self.state_names = self.config["state_names"]
        self.obs_dim = self.config["observation_dim"]
        
        self.transition_noise = self.config["transition"]["noise_scale"]
        self.emission_noise = self.config["emission"]["noise_scale"]
        self.momentum = self.config["transition"]["momentum"]
        
        # Model parameters (to be learned)
        self.transition_matrices = None  # F_k for each regime k
        self.emission_matrices = None    # H_k for each regime k
        self.regime_transitions = None   # P(regime_t | regime_{t-1})
        self.regime_probs = None        # Current regime probabilities
        
        # State estimates
        self.state_mean = None
        self.state_cov = None
        
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fetch_features_for_training(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Fetch features for model training
        
        Args:
            since: Start date
            until: End date
            
        Returns:
            Tuple of (timestamps_df, observations_array)
        """
        # Fetch all features from different tables and join
        # This is a simplified version - production would use proper joins
        
        # Options features
        surface_query = """
            SELECT timestamp, level, skew, curvature, term_structure, 
                   pca_factor_1, pca_factor_2
            FROM features_options_surface
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """
        
        # Hedging pressure
        pressure_query = """
            SELECT timestamp, stabilization_index, acceleration_index
            FROM features_hedging_pressure
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """
        
        # Onchain (daily data, need to forward fill)
        onchain_query = """
            SELECT timestamp, supply_elasticity, forced_flow_index, liquidity_impulse
            FROM features_onchain_indices
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """
        
        # Divergence
        divergence_query = """
            SELECT timestamp, divergence_score
            FROM features_divergence
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """
        
        if since is None:
            since = datetime.now() - timedelta(days=180)
        if until is None:
            until = datetime.now()
        
        df_surface = db_client.query_to_dataframe(surface_query, (since, until))
        df_pressure = db_client.query_to_dataframe(pressure_query, (since, until))
        df_onchain = db_client.query_to_dataframe(onchain_query, (since, until))
        df_divergence = db_client.query_to_dataframe(divergence_query, (since, until))
        
        # Merge on timestamp (using surface as base since it's most frequent)
        if len(df_surface) == 0:
            raise ValueError("No surface features found for training")
        
        df = df_surface.copy()
        
        # Merge pressure
        if len(df_pressure) > 0:
            df = df.merge(df_pressure, on='timestamp', how='left')
        
        # Merge divergence
        if len(df_divergence) > 0:
            df = df.merge(df_divergence, on='timestamp', how='left')
        
        # For onchain (daily), merge with forward fill
        if len(df_onchain) > 0:
            df = df.merge(df_onchain, on='timestamp', how='left')
            df[['supply_elasticity', 'forced_flow_index', 'liquidity_impulse']] = \
                df[['supply_elasticity', 'forced_flow_index', 'liquidity_impulse']].fillna(method='ffill')
        
        # Drop rows with too many NaNs
        df = df.dropna(thresh=len(df.columns) * 0.5)
        
        # Extract observations (fill remaining NaNs with 0)
        obs_columns = [c for c in df.columns if c != 'timestamp']
        observations = df[obs_columns].fillna(0).values
        
        logger.info(f"Loaded {len(observations)} observations with {observations.shape[1]} features")
        
        return df[['timestamp']], observations
    
    def initialize_parameters(self, observations: np.ndarray):
        """Initialize model parameters using K-means clustering
        
        Args:
            observations: Observation array (T × obs_dim)
        """
        logger.info("Initializing model parameters")
        
        # Use K-means to initialize regimes
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        regime_labels = kmeans.fit_predict(observations)
        
        # Initialize transition matrices (simple momentum model)
        self.transition_matrices = []
        for _ in range(self.n_regimes):
            F = np.eye(self.state_dim) * self.momentum
            self.transition_matrices.append(F)
        
        # Initialize emission matrices (random projection from state to obs)
        self.emission_matrices = []
        for k in range(self.n_regimes):
            # Find observations in this regime
            regime_mask = regime_labels == k
            if np.sum(regime_mask) > 0:
                regime_obs = observations[regime_mask]
                # Use PCA-like projection
                H = np.random.randn(self.obs_dim, self.state_dim) * 0.1
                self.emission_matrices.append(H)
            else:
                H = np.random.randn(self.obs_dim, self.state_dim) * 0.1
                self.emission_matrices.append(H)
        
        # Initialize regime transition matrix (sticky regimes)
        self.regime_transitions = np.zeros((self.n_regimes, self.n_regimes))
        for i in range(self.n_regimes):
            self.regime_transitions[i, i] = 0.9  # Stay in same regime
            for j in range(self.n_regimes):
                if i != j:
                    self.regime_transitions[i, j] = 0.1 / (self.n_regimes - 1)
        
        # Initialize state
        self.state_mean = np.zeros(self.state_dim)
        self.state_cov = np.eye(self.state_dim)
        
        # Initialize regime probabilities
        regime_counts = np.bincount(regime_labels, minlength=self.n_regimes)
        self.regime_probs = regime_counts / len(regime_labels)
        
        logger.info(f"Initialized with regime distribution: {self.regime_probs}")
    
    def forward_step(
        self,
        observation: np.ndarray,
        prev_regime_probs: np.ndarray,
        prev_state_mean: np.ndarray,
        prev_state_cov: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Single forward filtering step
        
        Args:
            observation: Current observation vector
            prev_regime_probs: Previous regime probabilities
            prev_state_mean: Previous state mean
            prev_state_cov: Previous state covariance
            
        Returns:
            Tuple of (new_regime_probs, new_state_mean, new_state_cov, likelihoods)
        """
        likelihoods = np.zeros(self.n_regimes)
        filtered_states = []
        filtered_covs = []
        
        # For each regime, compute likelihood and filtered state
        for k in range(self.n_regimes):
            # Predict
            F = self.transition_matrices[k]
            predicted_state = F @ prev_state_mean
            predicted_cov = F @ prev_state_cov @ F.T + np.eye(self.state_dim) * self.transition_noise
            
            # Update
            H = self.emission_matrices[k]
            predicted_obs = H @ predicted_state
            innovation = observation - predicted_obs
            innovation_cov = H @ predicted_cov @ H.T + np.eye(self.obs_dim) * self.emission_noise
            
            # Kalman gain
            K = predicted_cov @ H.T @ np.linalg.inv(innovation_cov)
            
            # Filtered state
            filtered_state = predicted_state + K @ innovation
            filtered_cov = (np.eye(self.state_dim) - K @ H) @ predicted_cov
            
            filtered_states.append(filtered_state)
            filtered_covs.append(filtered_cov)
            
            # Likelihood
            try:
                likelihood = multivariate_normal.pdf(
                    innovation,
                    mean=np.zeros(self.obs_dim),
                    cov=innovation_cov
                )
                likelihoods[k] = likelihood
            except:
                likelihoods[k] = 1e-10
        
        # Update regime probabilities using Bayes rule
        # P(regime_t | obs_t) ∝ P(obs_t | regime_t) * P(regime_t | regime_{t-1})
        predicted_regime_probs = self.regime_transitions.T @ prev_regime_probs
        posterior_regime_probs = likelihoods * predicted_regime_probs
        posterior_regime_probs = posterior_regime_probs / (np.sum(posterior_regime_probs) + 1e-10)
        
        # Mixture of filtered states
        new_state_mean = sum(p * s for p, s in zip(posterior_regime_probs, filtered_states))
        new_state_cov = sum(p * c for p, c in zip(posterior_regime_probs, filtered_covs))
        
        return posterior_regime_probs, new_state_mean, new_state_cov, likelihoods
    
    def fit(
        self,
        observations: np.ndarray,
        max_iterations: int = None,
        convergence_tol: float = None
    ):
        """Fit model using EM algorithm (simplified)
        
        Args:
            observations: Observation array (T × obs_dim)
            max_iterations: Maximum EM iterations
            convergence_tol: Convergence tolerance
        """
        if max_iterations is None:
            max_iterations = self.config["training"]["max_iterations"]
        if convergence_tol is None:
            convergence_tol = self.config["training"]["convergence_tol"]
        
        logger.info(f"Fitting Switching Kalman Filter with {len(observations)} observations")
        
        # Standardize observations
        observations_std = self.scaler.fit_transform(observations)
        
        # Initialize parameters
        self.initialize_parameters(observations_std)
        
        # Simplified training: just run forward pass (full EM would require backward pass)
        # This is a "filtering-only" approach suitable for online inference
        
        logger.info("Running forward pass for parameter estimation")
        
        regime_history = []
        state_history = []
        
        current_regime_probs = self.regime_probs.copy()
        current_state_mean = self.state_mean.copy()
        current_state_cov = self.state_cov.copy()
        
        for t, obs in enumerate(observations_std):
            regime_probs, state_mean, state_cov, _ = self.forward_step(
                obs,
                current_regime_probs,
                current_state_mean,
                current_state_cov
            )
            
            regime_history.append(regime_probs)
            state_history.append(state_mean)
            
            current_regime_probs = regime_probs
            current_state_mean = state_mean
            current_state_cov = state_cov
        
        self.regime_probs = current_regime_probs
        self.state_mean = current_state_mean
        self.state_cov = current_state_cov
        
        self.is_fitted = True
        
        logger.info(f"Model fitted. Final regime probs: {self.regime_probs}")
        logger.info(f"Final state mean: {self.state_mean}")
    
    def predict_regime(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict regime probabilities for new observation
        
        Args:
            observation: New observation vector
            
        Returns:
            Tuple of (regime_probs, state_mean)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Standardize observation
        obs_std = self.scaler.transform(observation.reshape(1, -1))[0]
        
        # Forward step
        regime_probs, state_mean, state_cov, _ = self.forward_step(
            obs_std,
            self.regime_probs,
            self.state_mean,
            self.state_cov
        )
        
        # Update internal state
        self.regime_probs = regime_probs
        self.state_mean = state_mean
        self.state_cov = state_cov
        
        return regime_probs, state_mean


def train_and_save_model(
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    model_version: str = None
) -> Dict:
    """Train state-space model and save to database
    
    Args:
        since: Training start date
        until: Training end date
        model_version: Model version string
        
    Returns:
        Dictionary with training results
    """
    if model_version is None:
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"Training state-space model version {model_version}")
    
    # Initialize model
    model = SwitchingKalmanFilter()
    
    # Fetch training data
    timestamps_df, observations = model.fetch_features_for_training(since, until)
    
    if len(observations) < 100:
        raise ValueError(f"Insufficient training data: {len(observations)} observations")
    
    # Fit model
    model.fit(observations)
    
    # Generate predictions for all training data
    results = []
    for i, (_, row) in enumerate(timestamps_df.iterrows()):
        timestamp = row['timestamp']
        obs = observations[i]
        
        regime_probs, state_mean = model.predict_regime(obs)
        
        record = {
            'timestamp': timestamp,
            'regime_1_prob': regime_probs[0],
            'regime_2_prob': regime_probs[1],
            'regime_3_prob': regime_probs[2],
            'state_risk_appetite': state_mean[0] if len(state_mean) > 0 else None,
            'state_leverage_stress': state_mean[1] if len(state_mean) > 1 else None,
            'state_dealer_stabilization': state_mean[2] if len(state_mean) > 2 else None,
            'state_tail_demand': state_mean[3] if len(state_mean) > 3 else None,
            'state_inventory_imbalance': state_mean[4] if len(state_mean) > 4 else None,
            'state_liquidity_regime': state_mean[5] if len(state_mean) > 5 else None,
            'model_version': model_version
        }
        results.append(record)
    
    # Save to database
    df_results = pd.DataFrame(results)
    db_client.insert_dataframe("model_states", df_results, if_exists="append")
    
    logger.info(f"Saved {len(results)} state estimates to database")
    
    return {
        'model_version': model_version,
        'n_observations': len(observations),
        'final_regime_probs': model.regime_probs.tolist(),
        'regime_names': model.regime_names
    }
