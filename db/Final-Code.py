
Kondlapudi Lasya <lasyareddy808@gmail.com>
4:15 PM (2 minutes ago)
to lasyareddy8804

# =====================================================================
# CRYPTOCURRENCY PORTFOLIO MANAGEMENT PROJECT
# =====================================================================
# This script combines various modules for cryptocurrency data processing,
# portfolio risk analysis, prediction, and stress testing into a single
# project structure.
# =====================================================================

import pandas as pd
import numpy as np
import sqlite3
import smtplib
from email.mime.text import MIMEText
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import time # Added for parallel_execution simulation

# =====================================================================
# 1. CRYPTOCURRENCY DATA PROCESSING (from Milestone 1)
# =====================================================================
print("--- 1. Cryptocurrency Data Processing ---")

def load_data(file_name, crypto_name):
    """
    Loads each CSV file, keeps only 'date' and 'close' columns,
    renames 'close' to the crypto name.
    """
    df = pd.read_csv(file_name)
    df['date'] = pd.to_datetime(df['date'])
    df = df[['date', 'close']]
    df.rename(columns={'close': crypto_name}, inplace=True)
    print(f"{crypto_name} data loaded successfully with {len(df)} records.")
    return df

file_names = {
"BTC": "Binance_BTCUSDT_d.csv",
"ETH": "Binance_ETHUSDT_d.csv",
"USDC": "Binance_USDCUSDT_d.csv"
}

print("Loading cryptocurrency data files in parallel...\n")
with ThreadPoolExecutor() as executor:
    results = list(executor.map(lambda item: load_data(item[1], item[0]), file_names.items()))

data = results[0]
for df in results[1:]:
    data = pd.merge(data, df, on='date', how='inner')
data.set_index('date', inplace=True)
print("\n=== Combined Cryptocurrency Data (First 10 Rows) ===")
print(data.head(10))
print("\nData Columns:", list(data.columns))

print("\nStoring combined data into SQLite database...")
conn_crypto = sqlite3.connect("crypto_data.db")
data.to_sql("Crypto_Prices", conn_crypto, if_exists="replace", index=True, index_label="Date")

def calculate_metrics(df, col_name):
    """
    Calculates key statistical metrics for each cryptocurrency:
    Mean, Standard Deviation, Max, Min, and Total Days.
    """
    series = df[col_name]
    metrics = {
        "Currency": col_name,
        "Mean Price": series.mean(),
        "Standard Deviation": series.std(),
        "Maximum Price": series.max(),
        "Minimum Price": series.min(),
        "Total Days": len(series)
    }
    print(f"Metrics calculated for {col_name}")
    return metrics

print("\nCalculating statistical metrics for each cryptocurrency in parallel...\n")
with ThreadPoolExecutor() as executor:
    metrics = list(executor.map(lambda col: calculate_metrics(data, col), data.columns))

metrics_df = pd.DataFrame(metrics)
print("\n=== Cryptocurrency Metrics DataFrame ===")
print(metrics_df)

print("\nStoring calculated metrics into the database...")
metrics_df.to_sql("Crypto_Metrics", conn_crypto, if_exists="replace", index=False)
print("Metrics stored successfully in 'crypto_data.db' (Table: Crypto_Metrics)")

print("\nMetrics also saved locally to 'crypto_metrics.csv'")
metrics_df.to_csv("crypto_metrics.csv", index=False)

conn_crypto.close()
print("\n--- Cryptocurrency Data Processing Complete ---")

# =====================================================================
# 2. PORTFOLIO MATH (Integrated)
# =====================================================================
print("\n--- 2. Portfolio Math ---")

# --- Example portfolio data (replace with your DB fetch later)
# Using data loaded from step 1 for consistency
assets = data.columns.tolist()
# Example returns and weights (replace with calculated values from metrics or rules)
# For demonstration, using sample data
returns = metrics_df['Mean Price'].values / metrics_df['Mean Price'].values.mean() # Example: proportional to mean price
weights = np.array([1/len(assets)] * len(assets)) # Equal weights for demonstration

# --- Check weights sum to 1
if not np.isclose(weights.sum(), 1.0):
    weights = weights / weights.sum() # Normalize weights

# --- Calculate covariance matrix from real_returns (from Stress Test section)
# Assuming real_returns is available from the later section
try:
    # Need to ensure real_returns is calculated before this section runs if needed
    # For now, using a placeholder if real_returns is not yet defined
    if 'real_returns' in locals():
         cov_matrix = real_returns.cov() * 252 # Annualize covariance
    else:
         # Placeholder covariance matrix if real_returns is not available
         cov_matrix = np.array([
             [0.04, 0.006, 0.004],
             [0.006, 0.03, 0.005],
             [0.004, 0.005, 0.02]
         ])
         print("Warning: real_returns not available, using placeholder covariance matrix.")

    # --- Portfolio return
    portfolio_return = np.dot(weights, returns)

    # --- Portfolio risk (std deviation)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_risk = np.sqrt(portfolio_variance)

    # --- Put results into DataFrame
    df_portfolio_math = pd.DataFrame({
        'Asset': assets,
        'Weight': weights,
        'ExpectedReturn': returns
    })

    print("Portfolio Data:\n", df_portfolio_math)
    print("\nPortfolio Expected Return:", round(portfolio_return, 4))
    print("Portfolio Risk (Std Dev):", round(portfolio_risk, 4))

except Exception as e:
    print(f"Error during Portfolio Math calculation: {e}")


print("\n--- Portfolio Math Complete ---")


# =====================================================================
# 3. DATABASE SETUP AND SAMPLE DATA (from db_portfolio.py)
# =====================================================================
print("\n--- 3. Database Setup and Sample Data ---")

conn_portfolio = sqlite3.connect("portfolio.db")
cursor = conn_portfolio.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS portfolio (
portfolio_id INTEGER PRIMARY KEY AUTOINCREMENT,
name TEXT,
total_value REAL,
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS portfolio_assets (
    asset_id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id INTEGER,
    currency TEXT,
    weight REAL,
    return REAL,
    risk REAL,
    metrics TEXT,
    FOREIGN KEY (portfolio_id) REFERENCES portfolio(portfolio_id)
)
""")

conn_portfolio.commit()

# --- Insert sample data (replace with actual calculated values later)
try:
    # Clear existing sample data for fresh run
    cursor.execute("DELETE FROM portfolio")
    cursor.execute("DELETE FROM portfolio_assets")
    conn_portfolio.commit()

    cursor.execute("INSERT INTO portfolio (name,total_value) VALUES (?,?)",
                   ("My Portfolio", 100000))
    portfolio_id = cursor.lastrowid

    # Using calculated metrics and weights from previous steps
    assets_data = []
    for i, asset in enumerate(assets):
        # Find corresponding metrics
        metric = metrics_df[metrics_df['Currency'] == asset].iloc[0]
        asset_return = returns[i] # Using the example returns calculated earlier
        # Using Standard Deviation as a proxy for risk for sample data
        asset_risk = metric['Standard Deviation']
        asset_weight = weights[i]

        assets_data.append((portfolio_id, asset, asset_weight, asset_return, asset_risk, 'Calculated Metrics'))

    cursor.executemany("""
    INSERT INTO portfolio_assets (portfolio_id, currency, weight, return, risk, metrics)
    VALUES (?,?,?,?,?,?)
    """, assets_data)
    conn_portfolio.commit()

    df_portfolio = pd.read_sql_query("SELECT * FROM portfolio", conn_portfolio)
    df_assets = pd.read_sql_query("SELECT * FROM portfolio_assets", conn_portfolio)

    print("Portfolio Table:\n", df_portfolio)
    print("\nPortfolio Assets Table:\n", df_assets)

except Exception as e:
    print(f"Error during Database Setup and Sample Data insertion: {e}")

conn_portfolio.close()
print("\n--- Database Setup and Sample Data Complete ---")


# =====================================================================
# 4. PARALLEL EXECUTION EXAMPLE (from parallel_execution.py)
# =====================================================================
print("\n--- 4. Parallel Execution Example ---")

def rule_equal_weight():
    time.sleep(1) # simulate calculation
    return "Equal-weight rule executed"

def rule_risk_based():
    time.sleep(1)
    return "Risk-based rule executed"

def rule_performance_based():
    time.sleep(1)
    return "Performance-based rule executed"

rules_example = [rule_equal_weight, rule_risk_based, rule_performance_based]

results_parallel = []
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(rule) for rule in rules_example]
    for f in futures:
        results_parallel.append(f.result())

print("Parallel Execution Results:")
for r in results_parallel:
    print("-", r)

print("\n--- Parallel Execution Example Complete ---")


# =====================================================================
# 5. COMPARE AND EXPORT EXAMPLE (from compare_and_export.py)
# =====================================================================
print("\n--- 5. Compare and Export Example ---")

# --- Sample Data (replace with actual portfolio and asset returns)
# Using portfolio_returns_full and one asset's return from Step 1 data
try:
    if 'portfolio_returns_full' in locals() and not real_returns.empty:
        df_compare = pd.DataFrame({
            'Date': portfolio_returns_full.index,
            'PortfolioReturn': portfolio_returns_full.values,
            'SingleAssetReturn': real_returns[assets[0]].values # Using the first asset for comparison
        })
        df_compare['Date'] = pd.to_datetime(df_compare['Date'])
    else:
         # Fallback to synthetic data if actual data is not available
        data_compare = {
            'Date': pd.date_range(start='2025-01-01', periods=10, freq='D'),
            'PortfolioReturn': [0.02, 0.01, -0.005, 0.03, 0.015, -0.01, 0.02, 0.025, 0.005, 0.03],
            'SingleAssetReturn': [0.03, 0.015, -0.01, 0.02, 0.02, -0.015, 0.018, 0.02, 0.0, 0.025]
        }
        df_compare = pd.DataFrame(data_compare)
        print("Warning: Actual return data not available, using synthetic data for comparison plot.")


    # --- Plot ---
    plt.figure(figsize=(8,5))
    plt.plot(df_compare['Date'], df_compare['PortfolioReturn'], label='Portfolio')
    plt.plot(df_compare['Date'], df_compare['SingleAssetReturn'], label='Single Asset')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.title('Portfolio vs Single Asset Return (Example)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Export to CSV ---
    df_compare.to_csv('portfolio_comparison.csv', index=False)
    print("Data exported to portfolio_comparison.csv")

except Exception as e:
    print(f"Error during Compare and Export example: {e}")

print("\n--- Compare and Export Example Complete ---")


# =====================================================================
# 6. RISK ANALYSIS AND EMAIL ALERT (from VGHhATKC38rq)
# =====================================================================
print("\n--- 6. Risk Analysis and Email Alert ---")

# Using real_returns from Stress Test section for risk metrics calculation
try:
    if 'real_returns' in locals() and not real_returns.empty and 'weights' in locals():

        # Using the first set of weights calculated in Portfolio Math (equal weights example)
        portfolio_rets_risk = (real_returns * weights).sum(axis=1)

        def annualized_volatility(daily_rets):
            return np.std(daily_rets, ddof=1) * np.sqrt(252)

        def sharpe_ratio(daily_rets, rf=0.03):
            excess = daily_rets - (rf / 252)
            return np.mean(excess) / np.std(daily_rets, ddof=1) * np.sqrt(252)

        def max_drawdown(daily_rets):
            cum = np.cumprod(1 + daily_rets)
            highwater = np.maximum.accumulate(cum)
            drawdowns = (cum - highwater) / highwater
            return -np.min(drawdowns)

        def sortino_ratio(daily_rets, rf=0.03):
            excess = daily_rets - (rf / 252)
            downside = excess[excess < 0]
            if len(downside) == 0:
                return np.inf
            return np.mean(excess) / np.std(downside, ddof=1) * np.sqrt(252)

        # Need market returns for Beta calculation. Using BTC as a proxy for market for simplicity
        if assets and assets[0] in real_returns.columns:
             market_returns_risk = real_returns[assets[0]].values
        else:
             # Fallback to synthetic market returns if real_returns is not available or assets list is empty
             np.random.seed(42)
             market_returns_risk = np.random.normal(0.07/252, 0.15/np.sqrt(252), len(real_returns))
             print("Warning: real_returns or assets list not available, using synthetic market returns for Beta.")


        def beta(portfolio_rets, market_rets):
            if len(portfolio_rets) != len(market_rets):
                 # Handle length mismatch if necessary, or ensure they are aligned
                 min_len = min(len(portfolio_rets), len(market_rets))
                 portfolio_rets = portfolio_rets[:min_len]
                 market_rets = market_rets[:min_len]
                 print("Warning: Portfolio and Market returns length mismatch, truncating to shortest length for Beta calculation.")

            if np.var(market_rets) == 0:
                return np.nan # Avoid division by zero
            cov = np.cov(portfolio_rets, market_rets)[0, 1]
            var = np.var(market_rets)
            return cov / var


        def max_asset_weight(weights):
            return np.max(weights)

        rules = {
            "Volatility ≤ 5%": annualized_volatility(portfolio_rets_risk) <= 0.05,
            "Sharpe Ratio ≥ 1": sharpe_ratio(portfolio_rets_risk) >= 1,
            "Max Drawdown ≥ -20%": max_drawdown(portfolio_rets_risk) >= -0.20,
            "Sortino Ratio ≥ 1": sortino_ratio(portfolio_rets_risk) >= 1,
            "Beta ≤ 1.2": beta(portfolio_rets_risk, market_returns_risk) <= 1.2 if len(portfolio_rets_risk) > 0 and len(market_returns_risk) > 0 else False, # Add check for empty data
            "Max Asset Weight ≤ 40%": max_asset_weight(weights) <= 0.40,
        }

        conn_risk = sqlite3.connect("risk_results.db")
        cursor_risk = conn_risk.cursor()
        cursor_risk.execute("""
        CREATE TABLE IF NOT EXISTS risk_results (
            rule TEXT,
            passed BOOLEAN
        )
        """)

        # Clear existing data
        cursor_risk.execute("DELETE FROM risk_results")
        conn_risk.commit()

        cursor_risk.executemany(
            "INSERT INTO risk_results (rule, passed) VALUES (?, ?)",
            [(rule, passed) for rule, passed in rules.items()]
        )
        conn_risk.commit()
        conn_risk.close()

        print("\n--- Risk Rule Results ---")
        for rule, passed in rules.items():
            print(f"{rule}: {'PASS' if passed else 'FAIL'}")

        # ----------------------------
        # Always Send Email
        # ----------------------------
        sender_email = "chanduchatopadyaya@gmail.com"
        receiver_email = "chanduchatopadyaya@gmail.com"
        # Replace with your Gmail App Password (16 characters)
        sender_pass = "hbid vyiv hblr tgjq" # Ensure this is securely handled

        msg_body = "\n".join([f"{rule}: {'PASS' if passed else 'FAIL'}" for rule, passed in rules.items()])
        msg = MIMEText(msg_body)
        msg["Subject"] = "RISK ALERT: Portfolio Check"
        msg["From"] = sender_email
        msg["To"] = receiver_email

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(sender_email, sender_pass)
                server.send_message(msg)
                print("\nEmail sent successfully")
        except smtplib.SMTPAuthenticationError:
            print("\nSMTP Authentication Error: Check your App Password and 2FA settings.")
        except Exception as e:
            print(f"\nFailed to send email: {e}")

    else:
        print("Skipping Risk Analysis and Email Alert: real_returns or weights data not available.")

except Exception as e:
    print(f"Error during Risk Analysis and Email Alert: {e}")

print("\n--- Risk Analysis and Email Alert Complete ---")


# =====================================================================
# 7. PREDICTOR MODULE (from predictor_xrp.py)
# =====================================================================
print("\n--- 7. Predictor Module ---")

def load_data_from_csvs_predictor():
    """Load Binance or Portfolio CSVs. If not found, generate synthetic demo data."""
    files = {
        "BTC": Path("Binance_BTCUSDT_d.csv"),
        "ETH": Path("Binance_ETHUSDT_d.csv"),
        "USDC": Path("Binance_USDCUSDT_d.csv"), # Changed from XRP to USDC to match available files
    }
    df = None
    found = False

    for k, p in files.items():
        if p.exists():
            try:
                tmp = pd.read_csv(p)
                cols_lower = [c.lower() for c in tmp.columns]
                if "close" in cols_lower:
                    col = tmp.columns[cols_lower.index("close")]
                    prices = pd.to_numeric(tmp[col], errors="coerce").ffill().fillna(0) # Using ffill directly
                else:
                    numeric_cols = tmp.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        prices = pd.to_numeric(tmp[numeric_cols[0]],
    errors="coerce").ffill().fillna(0) # Using ffill directly
                    else:
                        continue
                # Calculate daily returns (percentage change)
                series = prices.pct_change().fillna(0)
                colname = f"{k}_pct_change"
                if df is None:
                    df = pd.DataFrame({colname: series})
                else:
                    df[colname] = series
                found = True
            except Exception as e:
                print(f"Error loading or processing {p}: {e}")
                continue

    # Portfolio file check (using the comparison file generated earlier)
    portfolio_paths = [
        Path("portfolio_comparison.csv"),
    ]
    for p in portfolio_paths:
        if p.exists():
            try:
                tmp = pd.read_csv(p)
                # Look for a column that contains 'portfolio' and 'return' or 'change'
                candidates = [c for c in tmp.columns if "portfolio" in c.lower() and ("return" in c.lower() or "change" in c.lower())]
                if candidates:
                    col = candidates[0]
                    # Assuming this column already represents percentage change or can be converted
                    df["Portfolio_pct_change"] = pd.to_numeric(tmp[col], errors="coerce").fillna(0)
                    found = True
            except Exception as e:
                print(f"Error loading or processing {p}: {e}")
                continue


    if not found or df is None or df.empty:
        print("Warning: No suitable CSV data found, generating synthetic data for predictor.")
        rng = np.random.RandomState(42)
        n = 365 # Match the length of crypto data
        df = pd.DataFrame({
            "BTC_pct_change": rng.normal(0, 0.02, size=n).cumsum(), # Using more realistic return scales
            "ETH_pct_change": rng.normal(0, 0.025, size=n).cumsum(),
            "USDC_pct_change": rng.normal(0, 0.001, size=n).cumsum(), # USDC expected to have lower volatility
        })
        # Recalculate Portfolio_pct_change based on sy
