"""
Risk Calculation Engine (Monte Carlo VaR & Portfolio Optimization)
Calculates optimal portfolio weights and simulates future PnL.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os

# Basic Model Parameters
INITIAL_INVESTMENT = 100000  # Initial Capital
NUM_SIMULATIONS = 10000      # Number of Simulations
CONFIDENCE_LEVEL = 0.05      # Confidence Interval

def load_returns_data(): # Function that loads the Data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'mag7_returns.csv')
    returns_df = pd.read_csv(data_path, index_col=0, parse_dates=True)#Date_indexing
    return returns_df

def optimize_portfolio(returns_df):#PORTFOLIO OPTIMISATION
    print("Portfolio Optimization")
    num_assets = len(returns_df.columns)
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    
    #We minimise Negative Sharpe Ratio instead of Maximizing Positive Sharpe Ratio
    #Because of Library constraints 
    def calculate_negative_sharpe(weights):
        p_ret = np.sum(mean_returns * weights) * 252 # Ετησιοποιημένη απόδοση
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        return - (p_ret / p_vol)
    def check_weights_sum_to_one(weights):
        return np.sum(weights) - 1.0
    constraints = ({'type': 'eq', 'fun': check_weights_sum_to_one})
    #Bounds in Weights (0,100%)
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    #We start with equal weights as initial guess
    init_guess = num_assets * [1. / num_assets,]
    
    #The optimisation Function
    opt_results = minimize(calculate_negative_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return opt_results.x, mean_returns, cov_matrix

def run_monte_carlo_var(mean_daily_return, daily_volatility, initial_value=INITIAL_INVESTMENT):
    """
    Geometric Brownian Motion (GBM) Monte Carlo.
    """
    # N Z's between 0,1 normally distributed
    Z = np.random.normal(0, 1, NUM_SIMULATIONS)
    
    #Price(t+1) = Price(t)*e^((μ - (σ^2)/2 + σ*Z)
    drift = mean_daily_return - (0.5 * daily_volatility**2)
    shock = daily_volatility * Z
    simulated_end_values = initial_value * np.exp(drift + shock)
    
    # PnL
    pnl = simulated_end_values - initial_value
    
    # VaR
    var = np.percentile(pnl, CONFIDENCE_LEVEL * 100)
    
    # Conditional VaR (Expected Shortfall), Average Losses more than 95% in dist
    cvar = pnl[pnl <= var].mean()
    
    return pnl, var, cvar

if __name__ == "__main__":
    df = load_returns_data()
    
    weights, mean_returns, cov_matrix = optimize_portfolio(df)
    
    print("\n Optimal Portfolio Weights (Maximum Sharpe):")
    for ticker, weight in zip(df.columns, weights):
        print(f"{ticker}: {weight*100:.2f}%")
        
    #Portfolio Daily Return and Volatility
    port_daily_return = np.sum(mean_returns * weights)
    port_daily_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    #MonteCarlo Simulation
    pnl, var, cvar = run_monte_carlo_var(port_daily_return, port_daily_vol)
    
    print(f"\nMonte Carlo Results for 1-Day Horizon (Capital: ${INITIAL_INVESTMENT:,}):")
    print(f"Expected Daily Return: {port_daily_return*100:.3f}%")
    print(f"Daily Volatility (Risk): {port_daily_vol*100:.3f}%")
    print(f"Value at Risk (95%): {var:,.2f}$")
    print(f"Conditional VaR (CVaR): {cvar:,.2f}$")