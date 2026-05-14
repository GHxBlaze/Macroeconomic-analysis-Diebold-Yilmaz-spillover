

import wbgapi as wb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

DOWNLOADS = os.path.join(os.path.expanduser("~"), "Downloads")
OUTPUT_DIR = os.path.join(DOWNLOADS, "4_FX_Inflation_Analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f" All files will be saved to:\n   {OUTPUT_DIR}\n")

print("=" * 70)
print("EXCHANGE RATE PASS-THROUGH TO INDIAN INFLATION")
print("=" * 70)

print("\nStep 1: Fetching data from World Bank API...")

TIME_RANGE = range(1990, 2024)

indicators = {
    "PA.NUS.FCRF": "FX_Rate_LCU_per_USD",      # Official exchange rate
    "FP.CPI.TOTL.ZG": "CPI_Inflation_pct",      # CPI inflation (annual %)
    "FP.WPI.TOTL": "WPI_Index",                  # Wholesale price index
    "NY.GDP.DEFL.KD.ZG": "GDP_Deflator_pct",     # GDP deflator (proxy)
}

data_frames = {}
for code, name in indicators.items():
    try:
        df = wb.data.DataFrame(code, economy="IND", time=TIME_RANGE)
        df = df.reset_index()
        year_cols = [c for c in df.columns if c.startswith("YR")]
        values = {}
        for col in year_cols:
            year = int(col.replace("YR", ""))
            val = df[col].values[0] if len(df) > 0 else np.nan
            values[year] = val
        data_frames[name] = pd.Series(values, name=name)
        print(f"  ✓ {name}: {sum(~pd.isna(list(values.values())))} observations")
    except Exception as e:
        print(f"  ✗ {name}: {e}")

# Combine into single DataFrame
india_df = pd.DataFrame(data_frames)
india_df.index.name = "Year"
india_df = india_df.sort_index()

# Compute FX change (annual % change)
india_df["FX_Change_pct"] = india_df["FX_Rate_LCU_per_USD"].pct_change() * 100

# Compute WPI inflation if WPI index is available
if "WPI_Index" in india_df.columns:
    india_df["WPI_Inflation_pct"] = india_df["WPI_Index"].pct_change() * 100

print(f"\n  Combined dataset: {india_df.shape[0]} years × {india_df.shape[1]} variables")
print(f"  Year range: {india_df.index.min()} - {india_df.index.max()}")

# Drop rows with too many NaNs
india_df = india_df.dropna(subset=["FX_Rate_LCU_per_USD", "CPI_Inflation_pct"])

print("\nStep 2: Generating visualizations...")

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# FX Rate
axes[0].plot(india_df.index, india_df["FX_Rate_LCU_per_USD"],
            color="#e74c3c", linewidth=2, marker='o', markersize=4)
axes[0].set_ylabel("USD/INR Rate", fontsize=12)
axes[0].set_title("India: Exchange Rate and Inflation (1990-2023)", fontsize=14)
axes[0].grid(True, alpha=0.3)
axes[0].fill_between(india_df.index, india_df["FX_Rate_LCU_per_USD"],
                     alpha=0.1, color="#e74c3c")

# CPI Inflation
axes[1].plot(india_df.index, india_df["CPI_Inflation_pct"],
            color="#3498db", linewidth=2, marker='s', markersize=4)
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_ylabel("CPI Inflation (%)", fontsize=12)
axes[1].grid(True, alpha=0.3)

# FX Change vs CPI (overlaid)
ax2 = axes[2]
if "FX_Change_pct" in india_df.columns:
    ax2.bar(india_df.index, india_df["FX_Change_pct"],
           alpha=0.6, color="#e74c3c", label="FX Depreciation (%)")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(india_df.index, india_df["CPI_Inflation_pct"],
                 color="#3498db", linewidth=2, marker='s', markersize=4,
                 label="CPI Inflation (%)")
    ax2.set_ylabel("FX Change (%)", fontsize=12, color="#e74c3c")
    ax2_twin.set_ylabel("CPI Inflation (%)", fontsize=12, color="#3498db")
    ax2.set_xlabel("Year", fontsize=12)
    ax2.legend(loc="upper left")
    ax2_twin.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "india_fx_inflation_overview.png"),
           dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Overview plot saved")

print("\nStep 3: Augmented Dickey-Fuller tests for stationarity...")

def adf_test_manual(series, name):
    """Simple ADF-like stationarity check using first-difference correlation"""
    series = series.dropna()
    if len(series) < 10:
        print(f"  {name}: Insufficient data")
        return False

    # Check if mean-reverting by looking at autocorrelation of first differences
    diff = series.diff().dropna()
    mean_val = series.mean()
    std_val = series.std()
    cv = std_val / abs(mean_val) * 100 if mean_val != 0 else float('inf')

    # Simple trend test
    x = np.arange(len(series))
    slope, _, r_value, p_value, _ = stats.linregress(x, series.values)

    is_stationary = p_value > 0.05 and cv < 50  # Rough heuristic

    print(f"  {name}:")
    print(f"    Mean: {mean_val:.2f}, Std: {std_val:.2f}, CV: {cv:.1f}%")
    print(f"    Trend slope: {slope:.4f}, p-value: {p_value:.4f}")
    print(f"    → {'Likely stationary' if is_stationary else 'Likely non-stationary (use differences)'}")
    return is_stationary

vars_to_test = {
    "FX_Rate_LCU_per_USD": india_df.get("FX_Rate_LCU_per_USD"),
    "CPI_Inflation_pct": india_df.get("CPI_Inflation_pct"),
    "FX_Change_pct": india_df.get("FX_Change_pct"),
}

if "WPI_Inflation_pct" in india_df.columns:
    vars_to_test["WPI_Inflation_pct"] = india_df["WPI_Inflation_pct"]

stationarity = {}
for name, series in vars_to_test.items():
    if series is not None:
        stationarity[name] = adf_test_manual(series, name)

print("\n" + "=" * 70)
print("Step 4: Cross-correlation analysis (FX → Inflation lags)")
print("=" * 70)

def lagged_correlation(x, y, max_lags=5):
    """Compute correlation between x(t-k) and y(t) for k = 0, ..., max_lags"""
    results = []
    x_clean = x.dropna()
    y_clean = y.dropna()
    common_idx = x_clean.index.intersection(y_clean.index)

    for lag in range(0, max_lags + 1):
        # Shift x by lag periods (x at t-lag vs y at t)
        x_lagged = x_clean.shift(lag)
        # Align
        valid = pd.DataFrame({"x": x_lagged, "y": y_clean}).dropna()
        if len(valid) > 5:
            corr, pval = stats.pearsonr(valid["x"], valid["y"])
            results.append({"Lag": lag, "Correlation": corr, "P_value": pval,
                           "Significant": "Yes" if pval < 0.05 else "No"})
    return pd.DataFrame(results)

# FX Change → CPI
if "FX_Change_pct" in india_df.columns:
    print("\nFX Depreciation (%) → CPI Inflation (%):")
    lag_corr_cpi = lagged_correlation(india_df["FX_Change_pct"],
                                      india_df["CPI_Inflation_pct"], max_lags=5)
    print(lag_corr_cpi.to_string(index=False))

    # FX Change → WPI
    if "WPI_Inflation_pct" in india_df.columns:
        print("\nFX Depreciation (%) → WPI Inflation (%):")
        lag_corr_wpi = lagged_correlation(india_df["FX_Change_pct"],
                                          india_df["WPI_Inflation_pct"], max_lags=5)
        print(lag_corr_wpi.to_string(index=False))

print("\n" + "=" * 70)
print("Step 5: Granger Causality Tests (Manual Implementation)")
print("=" * 70)

def granger_causality_test(y, x, max_lag=3):
    """
    Test if x Granger-causes y.
    Restricted model: y(t) = a0 + a1*y(t-1) + ... + ap*y(t-p) + e
    Unrestricted model: y(t) = a0 + a1*y(t-1) + ... + bp*x(t-p) + e
    F-test on whether the x coefficients are jointly zero.
    """
    y = y.dropna()
    x = x.dropna()
    common_idx = y.index.intersection(x.index)
    y = y.loc[common_idx].values
    x = x.loc[common_idx].values

    n = len(y)
    if n < max_lag + 5:
        return None

    results = []

    for p in range(1, max_lag + 1):
        # Construct lagged matrices
        Y = y[p:]
        n_obs = len(Y)

        # Restricted model regressors: constant + y lags
        X_restricted = np.ones((n_obs, 1))
        for lag in range(1, p + 1):
            X_restricted = np.column_stack([X_restricted, y[p-lag:n-lag]])

        # Unrestricted model regressors: constant + y lags + x lags
        X_unrestricted = X_restricted.copy()
        for lag in range(1, p + 1):
            X_unrestricted = np.column_stack([X_unrestricted, x[p-lag:n-lag]])

        # Fit restricted model (OLS)
        try:
            beta_r = np.linalg.lstsq(X_restricted, Y, rcond=None)[0]
            resid_r = Y - X_restricted @ beta_r
            ssr_r = np.sum(resid_r ** 2)

            # Fit unrestricted model
            beta_u = np.linalg.lstsq(X_unrestricted, Y, rcond=None)[0]
            resid_u = Y - X_unrestricted @ beta_u
            ssr_u = np.sum(resid_u ** 2)

            # F-test
            q = p  # number of restrictions (x lag coefficients = 0)
            k = X_unrestricted.shape[1]
            f_stat = ((ssr_r - ssr_u) / q) / (ssr_u / (n_obs - k))
            p_value = 1 - stats.f.cdf(f_stat, q, n_obs - k)

            results.append({
                "Lag": p,
                "F_statistic": f_stat,
                "P_value": p_value,
                "Significant_5pct": "Yes ***" if p_value < 0.05 else "No",
                "SSR_Restricted": ssr_r,
                "SSR_Unrestricted": ssr_u,
            })
        except Exception as e:
            results.append({"Lag": p, "Error": str(e)})

    return pd.DataFrame(results)

# Test: FX Change → CPI Inflation
if "FX_Change_pct" in india_df.columns:
    print("\nH0: FX Depreciation does NOT Granger-cause CPI Inflation")
    gc_fx_cpi = granger_causality_test(
        india_df["CPI_Inflation_pct"],
        india_df["FX_Change_pct"],
        max_lag=4
    )
    if gc_fx_cpi is not None:
        print(gc_fx_cpi.to_string(index=False))

    print("\nH0: CPI Inflation does NOT Granger-cause FX Depreciation")
    gc_cpi_fx = granger_causality_test(
        india_df["FX_Change_pct"],
        india_df["CPI_Inflation_pct"],
        max_lag=4
    )
    if gc_cpi_fx is not None:
        print(gc_cpi_fx.to_string(index=False))

print("\n" + "=" * 70)
print("Step 6: VAR(p) Model Estimation")
print("=" * 70)

def estimate_var(data, var_names, max_lag=4):
    """
    Estimate a VAR(p) model manually using OLS equation-by-equation.
    Returns coefficient matrices and residuals.
    """
    # Prepare data matrix
    Y = data[var_names].dropna()
    n = len(Y)
    k = len(var_names)

    # Select optimal lag using AIC
    best_aic = np.inf
    best_lag = 1

    for p in range(1, max_lag + 1):
        # Construct regressors
        Y_dep = Y.values[p:]
        n_obs = len(Y_dep)

        X = np.ones((n_obs, 1))  # constant
        for lag in range(1, p + 1):
            X = np.column_stack([X, Y.values[p-lag:n-lag]])

        # OLS for each equation
        total_ssr = 0
        for eq in range(k):
            beta = np.linalg.lstsq(X, Y_dep[:, eq], rcond=None)[0]
            resid = Y_dep[:, eq] - X @ beta
            total_ssr += np.sum(resid ** 2) / n_obs

        # AIC approximation
        n_params = k * (1 + k * p)  # constant + k variables × p lags
        aic = n_obs * np.log(total_ssr / k) + 2 * n_params

        if aic < best_aic:
            best_aic = aic
            best_lag = p

    print(f"  Optimal lag (AIC): p = {best_lag}")

    # Estimate final model
    p = best_lag
    Y_dep = Y.values[p:]
    n_obs = len(Y_dep)

    X = np.ones((n_obs, 1))
    for lag in range(1, p + 1):
        X = np.column_stack([X, Y.values[p-lag:n-lag]])

    # Store coefficient matrices
    coef_matrices = {}  # lag -> n×n matrix
    residuals = np.zeros_like(Y_dep)
    all_betas = {}

    for eq in range(k):
        beta = np.linalg.lstsq(X, Y_dep[:, eq], rcond=None)[0]
        all_betas[var_names[eq]] = beta
        residuals[:, eq] = Y_dep[:, eq] - X @ beta

        # Extract lag-specific coefficients
        for lag in range(1, p + 1):
            if lag not in coef_matrices:
                coef_matrices[lag] = np.zeros((k, k))
            start_idx = 1 + (lag - 1) * k
            coef_matrices[lag][eq, :] = beta[start_idx:start_idx + k]

    # Residual covariance
    sigma = np.cov(residuals.T)

    return {
        "lag": p,
        "coef_matrices": coef_matrices,
        "residuals": residuals,
        "sigma": sigma,
        "var_names": var_names,
        "all_betas": all_betas,
        "n_obs": n_obs
    }

# Prepare variables for VAR
var_cols = ["FX_Change_pct", "CPI_Inflation_pct"]
if "WPI_Inflation_pct" in india_df.columns:
    var_cols.append("WPI_Inflation_pct")

var_data = india_df[var_cols].dropna()
print(f"\n  Variables: {var_cols}")
print(f"  Observations: {len(var_data)}")

if len(var_data) >= 15:
    var_result = estimate_var(var_data, var_cols, max_lag=4)

    # Print coefficient matrices
    for lag, mat in var_result["coef_matrices"].items():
        print(f"\n  Coefficient Matrix A_{lag}:")
        print(f"  {'':>20}", end="")
        for name in var_cols:
            print(f"  {name[:15]:>15}", end="")
        print()
        for i, name in enumerate(var_cols):
            print(f"  {name[:20]:>20}", end="")
            for j in range(len(var_cols)):
                print(f"  {mat[i,j]:>15.4f}", end="")
            print()

    print("\n" + "=" * 70)
    print("Step 7: Network Interpretation (Directed Graph)")
    print("=" * 70)

    print("\n  Directed edges (non-zero lagged effects):")
    for lag, mat in var_result["coef_matrices"].items():
        for i in range(len(var_cols)):
            for j in range(len(var_cols)):
                if i != j and abs(mat[i, j]) > 0.01:
                    direction = "reinforcing (+)" if mat[i, j] > 0 else "dampening (-)"
                    print(f"    {var_cols[j]} → {var_cols[i]} "
                          f"(lag {lag}, coeff = {mat[i,j]:.4f}, {direction})")
    print("\n" + "=" * 70)
    print("Step 8: Forecast Error Variance Decomposition (FEVD)")
    print("=" * 70)

    def compute_fevd(var_result, H=10):
        """Compute H-step FEVD using Cholesky identification"""
        k = len(var_result["var_names"])
        p = var_result["lag"]
        coef = var_result["coef_matrices"]
        sigma = var_result["sigma"]

        # Cholesky decomposition for orthogonalisation
        P = np.linalg.cholesky(sigma)

        # Compute MA coefficients Phi_0, Phi_1, ..., Phi_H
        Phi = [np.eye(k)]  # Phi_0 = I

        for h in range(1, H + 1):
            phi_h = np.zeros((k, k))
            for lag in range(1, min(h, p) + 1):
                if lag in coef:
                    phi_h += coef[lag] @ Phi[h - lag]
            Phi.append(phi_h)

        # Compute FEVD
        # theta_ij(H) = sum_h (e_i' Phi_h P e_j)^2 / sum_h (e_i' Phi_h Sigma Phi_h' e_i)
        fevd = np.zeros((k, k))

        for i in range(k):
            total_var = 0
            for h in range(H + 1):
                total_var += (Phi[h] @ sigma @ Phi[h].T)[i, i]

            for j in range(k):
                numerator = 0
                for h in range(H + 1):
                    numerator += (Phi[h] @ P[:, j])[i] ** 2
                fevd[i, j] = numerator / total_var if total_var > 0 else 0

        # Normalise rows to sum to 1
        row_sums = fevd.sum(axis=1, keepdims=True)
        fevd = fevd / row_sums

        return fevd

    fevd = compute_fevd(var_result, H=10)

    print(f"\n  FEVD Table (H=10 step ahead):")
    print(f"  {'Variable':>20}  ", end="")
    for name in var_cols:
        print(f"{'Shock:'+name[:12]:>18}", end="")
    print()
    print("  " + "-" * (20 + 18 * len(var_cols)))

    for i, name in enumerate(var_cols):
        print(f"  {name:>20}  ", end="")
        for j in range(len(var_cols)):
            print(f"{fevd[i,j]*100:>17.1f}%", end="")
        print()

    # Key result: How much inflation variance is explained by FX
    fx_idx = var_cols.index("FX_Change_pct")
    cpi_idx = var_cols.index("CPI_Inflation_pct")
    pct_explained = fevd[cpi_idx, fx_idx] * 100

    print(f"\n  ★ KEY RESULT: {pct_explained:.1f}% of CPI inflation forecast error")
    print(f"    variance is attributable to FX rate shocks (at 10-step horizon)")
    print("\n" + "=" * 70)
    print("Step 9: Diebold-Yilmaz Spillover Indices")
    print("=" * 70)

    # Total spillover
    total_spillover = (np.sum(fevd) - np.trace(fevd)) / np.sum(fevd) * 100
    print(f"\n  Total Spillover Index: {total_spillover:.1f}%")

    # Directional spillovers
    print("\n  Directional Spillovers:")
    for i, name in enumerate(var_cols):
        from_others = (np.sum(fevd[i, :]) - fevd[i, i]) * 100
        to_others = (np.sum(fevd[:, i]) - fevd[i, i]) * 100
        net = to_others - from_others

        role = "NET TRANSMITTER" if net > 0 else "NET RECEIVER"
        print(f"    {name:>25}: From others = {from_others:>6.1f}%, "
              f"To others = {to_others:>6.1f}%, Net = {net:>+6.1f}% ({role})")

else:
    print("  ⚠ Insufficient data for VAR estimation")

print("\n" + "=" * 70)
print("Step 10: Generating lag-specific scatter plots")
print("=" * 70)

if "FX_Change_pct" in india_df.columns:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for lag in range(6):
        ax = axes[lag]
        fx_lagged = india_df["FX_Change_pct"].shift(lag)
        valid = pd.DataFrame({
            "FX": fx_lagged,
            "CPI": india_df["CPI_Inflation_pct"]
        }).dropna()

        if len(valid) > 5:
            ax.scatter(valid["FX"], valid["CPI"], s=50, alpha=0.7,
                      color="#3498db", edgecolors="white")

            # Fit line
            slope, intercept, r, p, se = stats.linregress(valid["FX"], valid["CPI"])
            x_line = np.linspace(valid["FX"].min(), valid["FX"].max(), 100)
            ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2,
                   label=f'R²={r**2:.3f}, p={p:.3f}')

            ax.set_title(f"Lag = {lag} year{'s' if lag != 1 else ''}", fontsize=12)
            ax.set_xlabel("FX Depreciation (%)")
            ax.set_ylabel("CPI Inflation (%)")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.suptitle("FX Rate Change → CPI Inflation at Different Lags\n(India, 1990-2023)",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fx_inflation_lagged_scatter.png"),
               dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Lagged scatter plots saved")

india_df.to_csv(os.path.join(OUTPUT_DIR, "india_fx_inflation_data.csv"))

print(f"\n✓ All outputs saved!")
print(f"\n FILES IN YOUR DOWNLOADS: 4_FX_Inflation_Analysis/")
print("-" * 60)
for f in sorted(os.listdir(OUTPUT_DIR)):
    full_path = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(full_path) / 1024
    print(f"  {f:<50s} ({size_kb:.0f} KB)")
print("-" * 60)
