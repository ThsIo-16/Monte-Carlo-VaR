# Market Risk & Portfolio Optimization Engine

## Description
A quantitative finance project that performs Markowitz Portfolio Optimization (Maximum Sharpe Ratio) and calculates the 1-Day Value at Risk (VaR) and Conditional VaR (CVaR) using a Geometric Brownian Motion (GBM) Monte Carlo simulation. The model automates data extraction from Yahoo Finance and analyzes historical data for the Magnificent 7 stocks.

## Project Structure
```text
Monte_Carlo_VaR/
│
├── data/                    # Auto-generated directory for fetched CSVs
│
├── scripts/               
│   ├── data_api.py          # Fetches historical market data (yfinance)
│   ├── risk_calculation.py  # Quant engine (Markowitz & Monte Carlo GBM)
│   └── dashboard.py         # Streamlit web application
│
├── .gitignore
├── requirements.txt
└── README.md
```

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone <>
   cd Monte_Carlo_VaR
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Fetch market data:**
   ```bash
   python scripts/data_api.py
   ```

4. **Launch the Dashboard:**
   ```bash
   streamlit run scripts/dashboard.py
   ```