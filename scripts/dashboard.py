"""
Streamlit Dashboard for Monte Carlo VaR
UI
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Import our Functions
from risk_calculation import load_returns_data, optimize_portfolio, run_monte_carlo_var, INITIAL_INVESTMENT

#Config
st.set_page_config(page_title="Market Risk Engine", layout="wide")

st.title("Portfolio Risk & Monte Carlo Simulation")
st.markdown("VaR Calculation for an Optimized Mag 7 Portfolio")

st.sidebar.header("Settings: ")
st.sidebar.write(f"**Initial Capital:** ${INITIAL_INVESTMENT:,}")
st.sidebar.write("**Monte Carlo Scenarios:** 10,000")
st.sidebar.write("**Confidence Level:** 95%")

#Button
if st.sidebar.button("Run the Simulation"):
    
    with st.spinner("Data Loading and Portfolio Optimization"):
        df = load_returns_data()
        
        #Optimal Weights Calculation
        weights, mean_returns, cov_matrix = optimize_portfolio(df)
        
        #Portfolio Return and Volatility , and Annualised Sharpe Ratio
        port_daily_return = np.sum(mean_returns * weights)
        port_daily_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        annualized_sharpe = (port_daily_return / port_daily_vol) * np.sqrt(252)
        #Run MonteCarlo
        pnl, var, cvar = run_monte_carlo_var(port_daily_return, port_daily_vol)

    st.success("Simulation Successful!")

    # 1: Portfolio Results
    st.header("Portfolio Allocation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Only >1% Weighted Stocks
        st.write("Maximum Sharpe Ratio Weights:")
        for ticker, weight in zip(df.columns, weights):
            if weight > 0.01: 
                st.write(f"- **{ticker}:** {weight*100:.1f}%")
                
    with col2:
        #KPIs
        st.metric("Expected Daily Return", f"{port_daily_return*100:.3f}%")
        st.metric("Daily Risk (Volatility)", f"{port_daily_vol*100:.3f}%")
        st.metric("Annualised Sharpe Ratio", f"{annualized_sharpe:.2f}")
    st.divider()

    # 2: VaR
    st.header("Monte Carlo Value at Risk (1-Day)")
    
    col3, col4 = st.columns(2)
    col3.metric("Value at Risk (95%)", f"${var:,.2f}", delta="Worse 5%", delta_color="inverse")
    col4.metric("Conditional VaR (CVaR)", f"${cvar:,.2f}", delta="Avg Tail Loss", delta_color="inverse")

    #Matplotlib Graph
    fig, ax = plt.subplots(figsize=(10, 5))
    
    #PnL histogram
    sns.histplot(pnl, bins=50, ax=ax, color='lightgray', edgecolor='black')
    
    #VaR and CVaR Points
    ax.axvline(var, color='red', linestyle='--', label=f'VaR (95%): ${var:,.0f}')
    ax.axvline(cvar, color='darkred', linestyle='-', label=f'CVaR: ${cvar:,.0f}')
    
    ax.set_title('(PnL) - 10.000 Scenarios')
    ax.set_xlabel('Profit / Loss ($)')
    ax.set_ylabel('N')
    ax.legend()
    
    st.pyplot(fig)