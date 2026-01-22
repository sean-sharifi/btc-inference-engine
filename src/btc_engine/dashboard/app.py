"""Streamlit Dashboard for BTC Options + Onchain Engine"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from btc_engine.database.client import db_client
from btc_engine.utils.config_loader import settings

st.set_page_config(
    page_title="BTC Options + Onchain Engine",
    page_icon="📊",
    layout="wide"
)

st.title("🔮 BTC Options + Onchain Inference Engine")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Controls")
    
    refresh = st.button("🔄 Refresh Data", use_container_width=True)
    
    st.markdown("---")
    st.header("Status")
    
    # Check data availability
    latest_deribit = db_client.get_latest_timestamp("raw_deribit_ticker_snapshots")
    latest_glassnode = db_client.get_latest_timestamp("raw_glassnode_metrics")
    
    if latest_deribit:
        st.success(f"Deribit: {latest_deribit.strftime('%Y-%m-%d %H:%M')}")
    else:
        st.error("No Deribit data")
    
    if latest_glassnode:
        st.success(f"Glassnode: {latest_glassnode.strftime('%Y-%m-%d')}")
    else:
        st.error("No Glassnode data")

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Regimes",
    "📈 Options Surface",
    "⚖️ Hedging Pressure",
    "⛓️ Onchain Mechanics",
    "🔀 Divergence",
    "🎯 Forecasts"
])

with tab1:
    st.header("Regime Analysis")
    
    query = """
        SELECT timestamp, regime_1_prob, regime_2_prob, regime_3_prob,
               state_risk_appetite, state_leverage_stress, state_dealer_stabilization
        FROM model_states
        ORDER BY timestamp DESC
        LIMIT 500
    """
    
    try:
        df_states = db_client.query_to_dataframe(query)
        
        if len(df_states) > 0:
            # Regime probabilities over time
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_states['timestamp'],
                y=df_states['regime_1_prob'],
                name='Risk On',
                mode='lines',
                stackgroup='one',
                fillcolor='rgba(0, 255, 0, 0.3)'
            ))
            
            fig.add_trace(go.Scatter(
                x=df_states['timestamp'],
                y=df_states['regime_2_prob'],
                name='Compression',
                mode='lines',
                stackgroup='one',
                fillcolor='rgba(255, 165, 0, 0.3)'
            ))
            
            fig.add_trace(go.Scatter(
                x=df_states['timestamp'],
                y=df_states['regime_3_prob'],
                name='Distress',
                mode='lines',
                stackgroup='one',
                fillcolor='rgba(255, 0, 0, 0.3)'
            ))
            
            fig.update_layout(
                title="Regime Probabilities Over Time",
                xaxis_title="Time",
                yaxis_title="Probability",
                hovermode='x unified',
                height=400,
                yaxis_range=[0, 1]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Latest regime
            latest = df_states.iloc[0]
            regime_probs = [latest['regime_1_prob'], latest['regime_2_prob'], latest['regime_3_prob']]
            regime_names = ['Risk On', 'Compression', 'Distress']
            dominant_idx = regime_probs.index(max(regime_probs))
            
            st.subheader(f"Current Regime: {regime_names[dominant_idx]} ({max(regime_probs)*100:.1f}%)")
            
            cols = st.columns(3)
            cols[0].metric("Risk On", f"{latest['regime_1_prob']*100:.1f}%")
            cols[1].metric("Compression", f"{latest['regime_2_prob']*100:.1f}%")
            cols[2].metric("Distress", f"{latest['regime_3_prob']*100:.1f}%")
            
            # Latent states
            st.subheader("Latent State Estimates")
            cols2 = st.columns(3)
            cols2[0].metric("Risk Appetite", f"{latest['state_risk_appetite']:.3f}")
            cols2[1].metric("Leverage Stress", f"{latest['state_leverage_stress']:.3f}")
            cols2[2].metric("Dealer Stabilization", f"{latest['state_dealer_stabilization']:.3f}")
            
        else:
            st.warning("No state data available. Run: `btc-engine train-model`")
            
    except Exception as e:
        st.error(f"Error loading state data: {e}")

with tab2:
    st.header("Options Surface Factors")
    
    # Fetch surface factors
    query = """
        SELECT timestamp, level, skew, curvature, term_structure, wing_asymmetry
        FROM features_options_surface
        ORDER BY timestamp DESC
        LIMIT 500
    """
    
    try:
        df_surface = db_client.query_to_dataframe(query)
        
        if len(df_surface) > 0:
            # Plot factors
            fig = go.Figure()
            
            for col in ['level', 'skew', 'curvature']:
                fig.add_trace(go.Scatter(
                    x=df_surface['timestamp'],
                    y=df_surface[col],
                    name=col.title(),
                    mode='lines'
                ))
            
            fig.update_layout(
                title="Options Surface Factors",
                xaxis_title="Time",
                yaxis_title="Value",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Latest values
            latest = df_surface.iloc[0]
            cols = st.columns(5)
            cols[0].metric("Level", f"{latest['level']:.4f}")
            cols[1].metric("Skew", f"{latest['skew']:.4f}")
            cols[2].metric("Curvature", f"{latest['curvature']:.4f}")
            cols[3].metric("Term", f"{latest['term_structure']:.4f}")
            cols[4].metric("Wing Asym", f"{latest['wing_asymmetry']:.4f}")
        else:
            st.warning("No surface data available. Run: `btc-engine build-features`")
            
    except Exception as e:
        st.error(f"Error loading surface data: {e}")

with tab3:
    st.header("Hedging Pressure")
    
    query = """
        SELECT timestamp, stabilization_index, acceleration_index, 
               max_gamma_strike, total_gamma_exposure
        FROM features_hedging_pressure
        ORDER BY timestamp DESC
        LIMIT 500
    """
    
    try:
        df_pressure = db_client.query_to_dataframe(query)
        
        if len(df_pressure) > 0:
            # Plot indices
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_pressure['timestamp'],
                y=df_pressure['stabilization_index'],
                name='Stabilization',
                mode='lines',
                line=dict(color='green')
            ))
            
            fig.add_trace(go.Scatter(
                x=df_pressure['timestamp'],
                y=df_pressure['acceleration_index'],
                name='Acceleration',
                mode='lines',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title="Hedging Pressure Indices",
                xaxis_title="Time",
                yaxis_title="Index Value",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Latest values
            latest = df_pressure.iloc[0]
            cols = st.columns(3)
            cols[0].metric("Stabilization", f"{latest['stabilization_index']:.4f}")
            cols[1].metric("Acceleration", f"{latest['acceleration_index']:.4f}")
            cols[2].metric("Max Gamma Strike", f"${latest['max_gamma_strike']:.0f}")
        else:
            st.warning("No pressure data available. Run: `btc-engine build-features`")
            
    except Exception as e:
        st.error(f"Error loading pressure data: {e}")

with tab4:
    st.header("Onchain Mechanics")
    
    query = """
        SELECT timestamp, supply_elasticity, forced_flow_index, liquidity_impulse
        FROM features_onchain_indices
        ORDER BY timestamp DESC
        LIMIT 500
    """
    
    try:
        df_onchain = db_client.query_to_dataframe(query)
        
        if len(df_onchain) > 0:
            # Plot indices
            fig = go.Figure()
            
            for col, color in [('supply_elasticity', 'blue'), ('forced_flow_index', 'orange'), ('liquidity_impulse', 'green')]:
                fig.add_trace(go.Scatter(
                    x=df_onchain['timestamp'],
                    y=df_onchain[col],
                    name=col.replace('_', ' ').title(),
                    mode='lines',
                    line=dict(color=color)
                ))
            
            fig.update_layout(
                title="Onchain Mechanical Indices",
                xaxis_title="Date",
                yaxis_title="Z-Score",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Latest values
            latest = df_onchain.iloc[0]
            cols = st.columns(3)
            cols[0].metric("Supply Elasticity", f"{latest['supply_elasticity']:.2f}")
            cols[1].metric("Forced Flow", f"{latest['forced_flow_index']:.2f}")
            cols[2].metric("Liquidity Impulse", f"{latest['liquidity_impulse']:.2f}")
        else:
            st.warning("No onchain data available. Run: `btc-engine build-features`")
            
    except Exception as e:
        st.error(f"Error loading onchain data: {e}")

with tab5:
    st.header("Divergence Scoreboard")
    
    query = """
        SELECT *
        FROM features_divergence
        ORDER BY timestamp DESC
        LIMIT 100
    """
    
    try:
        df_divergence = db_client.query_to_dataframe(query)
        
        if len(df_divergence) > 0:
            # Latest classification
            latest = df_divergence.iloc[0]
            
            st.subheader(f"Current: {latest['classification']}")
            st.metric("Divergence Score", f"{latest['divergence_score']:.2f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Options Tail Signal", f"{latest['options_tail_signal']:.2f}")
            with col2:
                st.metric("Onchain Pressure Signal", f"{latest['onchain_pressure_signal']:.2f}")
            
            # Top signals
            st.subheader("Top Contributing Signals")
            for i in range(1, 6):
                signal_col = f'top_signal_{i}'
                value_col = f'top_signal_{i}_value'
                if signal_col in latest and latest[signal_col]:
                    st.write(f"{i}. **{latest[signal_col]}**: {latest[value_col]:.4f}")
            
            # Time series
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_divergence['timestamp'],
                y=df_divergence['divergence_score'],
                name='Divergence Score',
                mode='lines+markers'
            ))
            
            fig.update_layout(
                title="Divergence Score Over Time",
                xaxis_title="Time",
                yaxis_title="Score",
                hovermode='x unified',
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No divergence data available. Run: `btc-engine build-features`")
            
    except Exception as e:
        st.error(f"Error loading divergence data: {e}")

with tab6:
    st.header("Distributional Forecasts")
    
    query = """
        SELECT forecast_timestamp, target_timestamp, horizon,
               quantile_05, quantile_25, quantile_50, quantile_75, quantile_95,
               expected_shortfall_5pct, vol_of_vol
        FROM forecasts
        ORDER BY forecast_timestamp DESC
        LIMIT 100
    """
    
    try:
        df_forecasts = db_client.query_to_dataframe(query)
        
        if len(df_forecasts) > 0:
            # Latest forecast
            latest_24h = df_forecasts[df_forecasts['horizon'] == '24h'].iloc[0] if len(df_forecasts[df_forecasts['horizon'] == '24h']) > 0 else None
            latest_7d = df_forecasts[df_forecasts['horizon'] == '7d'].iloc[0] if len(df_forecasts[df_forecasts['horizon'] == '7d']) > 0 else None
            
            if latest_24h is not None:
                st.subheader("24-Hour Forecast")
                
                # Fan chart
                fig = go.Figure()
                
                quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
                values = [latest_24h['quantile_05'], latest_24h['quantile_25'], 
                         latest_24h['quantile_50'], latest_24h['quantile_75'], latest_24h['quantile_95']]
                
                fig.add_trace(go.Bar(
                    x=['Q05', 'Q25', 'Q50', 'Q75', 'Q95'],
                    y=[v*100 for v in values],
                    marker_color=['red', 'orange', 'blue', 'lightgreen', 'green']
                ))
                
                fig.update_layout(
                    title="24h Return Quantiles (%)",
                    xaxis_title="Quantile",
                    yaxis_title="Return (%)",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                cols = st.columns(5)
                cols[0].metric("Q05", f"{latest_24h['quantile_05']*100:.2f}%")
                cols[1].metric("Q25", f"{latest_24h['quantile_25']*100:.2f}%")
                cols[2].metric("Q50", f"{latest_24h['quantile_50']*100:.2f}%")
                cols[3].metric("Q75", f"{latest_24h['quantile_75']*100:.2f}%")
                cols[4].metric("Q95", f"{latest_24h['quantile_95']*100:.2f}%")
                
                st.metric("Expected Shortfall (5%)", f"{latest_24h['expected_shortfall_5pct']*100:.2f}%")
            
            if latest_7d is not None:
                st.subheader("7-Day Forecast")
                
                cols = st.columns(5)
                cols[0].metric("Q05", f"{latest_7d['quantile_05']*100:.2f}%")
                cols[1].metric("Q25", f"{latest_7d['quantile_25']*100:.2f}%")
                cols[2].metric("Q50", f"{latest_7d['quantile_50']*100:.2f}%")
                cols[3].metric("Q75", f"{latest_7d['quantile_75']*100:.2f}%")
                cols[4].metric("Q95", f"{latest_7d['quantile_95']*100:.2f}%")
        else:
            st.warning("No forecast data available. Run: `btc-engine forecast`")
            
    except Exception as e:
        st.error(f"Error loading forecast data: {e}")

# Footer
st.markdown("---")
st.caption(f"Database: {settings.database_path} | Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
