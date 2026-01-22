"""Options ↔ Onchain divergence detection and scoring"""

from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from btc_engine.database.client import db_client
from btc_engine.utils.config_loader import load_yaml_config
from btc_engine.utils.logging_config import logger


class DivergenceDetector:
    """Detect divergence between options pricing and onchain fundamentals"""
    
    def __init__(self):
        """Initialize divergence detector"""
        config = load_yaml_config("features")
        self.config = config["divergence"]
        
        self.thresholds = self.config["threshold_scores"]
        self.top_n_signals = self.config["explainability"]["top_n_signals"]
        self.lookback_days = self.config["lookback_correlation_days"]
    
    def fetch_options_signals(self, timestamp: datetime) -> Dict[str, float]:
        """Fetch options-based signals
        
        Args:
            timestamp: Timestamp to fetch
            
        Returns:
            Dictionary with options signals
        """
        signals = {}
        
        # Surface factors
        surface_query = """
            SELECT skew, curvature, wing_asymmetry
            FROM features_options_surface
            WHERE timestamp = ?
        """
        surface_df = db_client.query_to_dataframe(surface_query, (timestamp,))
        
        if len(surface_df) > 0:
            signals['options_skew'] = surface_df['skew'].iloc[0]
            signals['options_curvature'] = surface_df['curvature'].iloc[0]
            signals['options_wing_asymmetry'] = surface_df['wing_asymmetry'].iloc[0]
        
        # Risk-neutral metrics (aggregate across expiries)
        rn_query = """
            SELECT 
                AVG(left_tail_mass) as avg_left_tail,
                AVG(right_tail_mass) as avg_right_tail,
                AVG(jump_risk_proxy) as avg_jump_risk
            FROM features_risk_neutral
            WHERE timestamp = ?
        """
        rn_df = db_client.query_to_dataframe(rn_query, (timestamp,))
        
        if len(rn_df) > 0:
            signals['options_left_tail_mass'] = rn_df['avg_left_tail'].iloc[0]
            signals['options_right_tail_mass'] = rn_df['avg_right_tail'].iloc[0]
            signals['options_jump_risk'] = rn_df['avg_jump_risk'].iloc[0]
        
        # Hedging pressure
        pressure_query = """
            SELECT stabilization_index, acceleration_index
            FROM features_hedging_pressure
            WHERE timestamp = ?
        """
        pressure_df = db_client.query_to_dataframe(pressure_query, (timestamp,))
        
        if len(pressure_df) > 0:
            signals['options_stabilization'] = pressure_df['stabilization_index'].iloc[0]
            signals['options_acceleration'] = pressure_df['acceleration_index'].iloc[0]
        
        return signals
    
    def fetch_onchain_signals(self, timestamp: datetime) -> Dict[str, float]:
        """Fetch onchain-based signals
        
        Args:
            timestamp: Timestamp to fetch (will use nearest available)
            
        Returns:
            Dictionary with onchain signals
        """
        signals = {}
        
        # Onchain indices (daily data, so get nearest)
        query = """
            SELECT *
            FROM features_onchain_indices
            WHERE timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT 1
        """
        df = db_client.query_to_dataframe(query, (timestamp,))
        
        if len(df) > 0:
            signals['onchain_elasticity'] = df['supply_elasticity'].iloc[0]
            signals['onchain_forced_flow'] = df['forced_flow_index'].iloc[0]
            signals['onchain_liquidity'] = df['liquidity_impulse'].iloc[0]
            signals['onchain_elasticity_z'] = df['elasticity_zscore'].iloc[0]
            signals['onchain_forced_flow_z'] = df['forced_flow_zscore'].iloc[0]
            signals['onchain_liquidity_z'] = df['liquidity_zscore'].iloc[0]
        
        return signals
    
    def calculate_tail_signal(self, options_signals: Dict[str, float]) -> float:
        """Calculate aggregate options tail stress signal
        
        Args:
            options_signals: Dictionary with options metrics
            
        Returns:
            Tail signal value (higher → more tail stress priced in)
        """
        components = []
        
        # Left tail mass (downside fear)
        if 'options_left_tail_mass' in options_signals:
            components.append(options_signals['options_left_tail_mass'] * 2.0)
        
        # Negative skew (downside demand)
        if 'options_skew' in options_signals:
            components.append(abs(min(0, options_signals['options_skew'])))
        
        # Jump risk
        if 'options_jump_risk' in options_signals:
            components.append(options_signals['options_jump_risk'])
        
        # Curvature (smile intensity)
        if 'options_curvature' in options_signals:
            components.append(abs(options_signals['options_curvature']))
        
        return np.mean(components) if components else 0.0
    
    def calculate_onchain_pressure_signal(self, onchain_signals: Dict[str, float]) -> float:
        """Calculate aggregate onchain selling pressure signal
        
        Args:
            onchain_signals: Dictionary with onchain metrics
            
        Returns:
            Pressure signal value (higher → more fundamental selling pressure)
        """
        components = []
        
        # Forced flow (positive → selling pressure)
        if 'onchain_forced_flow_z' in onchain_signals:
            components.append(max(0, onchain_signals['onchain_forced_flow_z']))
        
        # Low elasticity (harder to absorb selling)
        if 'onchain_elasticity_z' in onchain_signals:
            components.append(-min(0, onchain_signals['onchain_elasticity_z']))
        
        # Negative liquidity impulse (liquidity draining)
        if 'onchain_liquidity_z' in onchain_signals:
            components.append(-min(0, onchain_signals['onchain_liquidity_z']))
        
        return np.mean(components) if components else 0.0
    
    def classify_divergence(
        self,
        tail_signal: float,
        pressure_signal: float
    ) -> Tuple[str, float]:
        """Classify divergence type
        
        Args:
            tail_signal: Options tail stress signal
            pressure_signal: Onchain pressure signal
            
        Returns:
            Tuple of (classification, divergence_score)
        """
        # Divergence score: difference between signals
        divergence_score = tail_signal - pressure_signal
        
        # Classify based on thresholds
        if divergence_score > self.thresholds["hedge_demand"]:
            # High options demand but low onchain pressure → hedging/overpriced fear
            classification = "Hedge Demand / False Fear"
        
        elif divergence_score < self.thresholds["underpriced_crash"]:
            # Low options demand but high onchain pressure → underpriced risk
            classification = "Underpriced Crash Risk"
        
        elif tail_signal > 0 and pressure_signal < -0.5:
            # High call skew + low elasticity → breakout potential
            classification = "Supply Inelastic Breakout Risk"
        
        else:
            # Signals aligned
            classification = "Balanced / Consistent"
        
        return classification, divergence_score
    
    def rank_contributing_signals(
        self,
        options_signals: Dict[str, float],
        onchain_signals: Dict[str, float]
    ) -> List[Tuple[str, float]]:
        """Rank top contributing signals by magnitude
        
        Args:
            options_signals: Options signals dict
            onchain_signals: Onchain signals dict
            
        Returns:
            List of (signal_name, value) tuples sorted by importance
        """
        all_signals = {}
        all_signals.update(options_signals)
        all_signals.update(onchain_signals)
        
        # Sort by absolute value
        ranked = sorted(
            all_signals.items(),
            key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0,
            reverse=True
        )
        
        return ranked[:self.top_n_signals]
    
    def detect_divergence(self, timestamp: datetime) -> Dict[str, any]:
        """Detect and classify divergence
        
        Args:
            timestamp: Timestamp to analyze
            
        Returns:
            Dictionary with divergence metrics
        """
        logger.info(f"Detecting divergence for {timestamp}")
        
        # Fetch signals
        options_signals = self.fetch_options_signals(timestamp)
        onchain_signals = self.fetch_onchain_signals(timestamp)
        
        if not options_signals or not onchain_signals:
            logger.warning("Incomplete signals for divergence detection")
            return {
                'timestamp': timestamp,
                'divergence_score': np.nan,
                'classification': 'Insufficient Data',
                'options_tail_signal': np.nan,
                'onchain_pressure_signal': np.nan,
            }
        
        # Calculate aggregate signals
        tail_signal = self.calculate_tail_signal(options_signals)
        pressure_signal = self.calculate_onchain_pressure_signal(onchain_signals)
        
        # Classify
        classification, divergence_score = self.classify_divergence(tail_signal, pressure_signal)
        
        # Rank contributing signals
        top_signals = self.rank_contributing_signals(options_signals, onchain_signals)
        
        result = {
            'timestamp': timestamp,
            'divergence_score': divergence_score,
            'classification': classification,
            'options_tail_signal': tail_signal,
            'onchain_pressure_signal': pressure_signal,
        }
        
        # Add top 5 signals
        for i, (signal_name, signal_value) in enumerate(top_signals[:5], 1):
            result[f'top_signal_{i}'] = signal_name
            result[f'top_signal_{i}_value'] = signal_value
        
        # Fill remaining slots if less than 5
        for i in range(len(top_signals) + 1, 6):
            result[f'top_signal_{i}'] = None
            result[f'top_signal_{i}_value'] = np.nan
        
        logger.info(f"Divergence: {classification} (score={divergence_score:.2f})")
        
        return result


def calculate_and_store_divergence(timestamp: datetime) -> Dict[str, any]:
    """Calculate and store divergence metrics
    
    Args:
        timestamp: Timestamp to process
        
    Returns:
        Dictionary with divergence metrics
    """
    detector = DivergenceDetector()
    metrics = detector.detect_divergence(timestamp)
    
    # Store to database
    df = pd.DataFrame([metrics])
    db_client.insert_dataframe("features_divergence", df, if_exists="append")
    
    logger.info("Stored divergence metrics")
    
    return metrics
