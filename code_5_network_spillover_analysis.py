
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from itertools import product as iter_product
import os
import warnings
warnings.filterwarnings('ignore')

DOWNLOADS = os.path.join(os.path.expanduser("~"), "Downloads")
OUTPUT_DIR = os.path.join(DOWNLOADS, "5_Network_Spillover")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f" All files will be saved to:\n   {OUTPUT_DIR}\n")

np.random.seed(42)

print("=" * 70)
print("DIRECTED NETWORK FROM TIME SERIES")
print("VAR, Granger Causality, & Diebold-Yilmaz Spillover Analysis")
print("=" * 70)


print("\n1. GENERATING SIMULATED MACRO-FINANCIAL DATA")
print("-" * 50)

# Variables representing a simplified Indian macro-financial system
VAR_NAMES = ["FX_Rate", "Oil_Price", "CPI", "WPI", "Interest_Rate", "Nifty"]
n_vars = len(VAR_NAMES)
T = 300  # 300 monthly observations (~25 years)
p_true = 2  # True lag order

# True VAR(2) coefficient matrices
# Designed to reflect realistic economic relationships:
# - Oil → WPI (oil feeds into wholesale prices)
# - FX → CPI, WPI (exchange rate pass-through)
# - Interest_Rate → Nifty (monetary policy affects stocks)
# - CPI → Interest_Rate (Taylor rule)
# - Oil → FX (current account effects)

A1 = np.array([
    # FX      Oil     CPI     WPI     IntRate  Nifty
    [ 0.50,   0.05,   0.00,   0.00,   0.10,   0.00],  # FX equation
    [ 0.00,   0.60,   0.00,   0.00,   0.00,   0.00],  # Oil equation
    [ 0.15,   0.05,   0.40,   0.10,   0.00,   0.00],  # CPI equation
    [ 0.20,   0.15,   0.05,   0.35,   0.00,   0.00],  # WPI equation
    [ 0.00,   0.00,   0.20,   0.00,   0.30,  -0.05],  # Interest rate eq
    [ 0.05,  -0.05,   0.00,   0.00,  -0.15,   0.45],  # Nifty equation
])

A2 = np.array([
    [ 0.10,   0.03,   0.00,   0.00,   0.05,   0.00],
    [ 0.00,   0.15,   0.00,   0.00,   0.00,   0.00],
    [ 0.08,   0.03,   0.15,   0.05,   0.00,   0.00],
    [ 0.10,   0.08,   0.03,   0.10,   0.00,   0.00],
    [ 0.00,   0.00,   0.10,   0.00,   0.10,   0.00],
    [ 0.00,  -0.03,   0.00,   0.00,  -0.08,   0.15],
])

# Shock covariance (some correlation between shocks)
Sigma = np.eye(n_vars) * 0.5
Sigma[0, 1] = 0.1; Sigma[1, 0] = 0.1  # FX-Oil correlation
Sigma[2, 3] = 0.2; Sigma[3, 2] = 0.2  # CPI-WPI correlation

# Generate VAR(2) process
data = np.zeros((T + 100, n_vars))  # extra for burn-in
for t in range(2, T + 100):
    eps = np.random.multivariate_normal(np.zeros(n_vars), Sigma)
    data[t] = A1 @ data[t-1] + A2 @ data[t-2] + eps

# Discard burn-in
data = data[100:]
df = pd.DataFrame(data, columns=VAR_NAMES)

print(f"  Variables: {VAR_NAMES}")
print(f"  Observations: {T}")
print(f"  True lag order: {p_true}")
print(f"\n  Summary statistics:")
print(df.describe().round(3).to_string())

print("\n\n2. VAR ESTIMATION")
print("-" * 50)

def estimate_var_full(data, max_lag=6):
    """
    Estimate VAR(p) with optimal lag selection using AIC, BIC, HQ.
    Returns coefficient matrices, residuals, and diagnostics.
    """
    Y = data.values
    T, k = Y.shape

    # Lag selection
    criteria = []
    for p in range(1, max_lag + 1):
        Y_dep = Y[p:]
        n = len(Y_dep)

        # Construct regressors: [1, Y_{t-1}, ..., Y_{t-p}]
        X = np.ones((n, 1))
        for lag in range(1, p + 1):
            X = np.column_stack([X, Y[p-lag:T-lag]])

        # OLS equation-by-equation
        B = np.linalg.lstsq(X, Y_dep, rcond=None)[0]
        resid = Y_dep - X @ B
        Sigma_hat = (resid.T @ resid) / n

        # Information criteria
        log_det = np.log(np.linalg.det(Sigma_hat))
        n_params = k * (1 + k * p)
        aic = log_det + 2 * n_params / n
        bic = log_det + np.log(n) * n_params / n
        hq = log_det + 2 * np.log(np.log(n)) * n_params / n

        criteria.append({"Lag": p, "AIC": aic, "BIC": bic, "HQ": hq})

    criteria_df = pd.DataFrame(criteria)
    print("\n  Information Criteria:")
    print(criteria_df.to_string(index=False))

    best_p_aic = criteria_df.loc[criteria_df["AIC"].idxmin(), "Lag"]
    best_p_bic = criteria_df.loc[criteria_df["BIC"].idxmin(), "Lag"]
    print(f"\n  Selected lag: AIC → p={best_p_aic}, BIC → p={best_p_bic}")

    # Estimate with AIC-selected lag
    p = int(best_p_aic)
    Y_dep = Y[p:]
    n = len(Y_dep)

    X = np.ones((n, 1))
    for lag in range(1, p + 1):
        X = np.column_stack([X, Y[p-lag:T-lag]])

    B = np.linalg.lstsq(X, Y_dep, rcond=None)[0]
    resid = Y_dep - X @ B
    Sigma_hat = (resid.T @ resid) / n

    # Extract coefficient matrices
    coef_matrices = {}
    for lag in range(1, p + 1):
        start = 1 + (lag - 1) * k
        coef_matrices[lag] = B[start:start+k, :].T  # Transpose for row=equation

    return {
        "lag": p,
        "coef_matrices": coef_matrices,
        "intercept": B[0, :],
        "residuals": resid,
        "sigma": Sigma_hat,
        "var_names": list(data.columns),
        "n_obs": n,
        "criteria": criteria_df
    }

var_result = estimate_var_full(df, max_lag=5)
p = var_result["lag"]

print(f"\n  Estimated coefficient matrices:")
for lag, A in var_result["coef_matrices"].items():
    print(f"\n  A_{lag}:")
    print(f"  {'':>14}", end="")
    for name in VAR_NAMES:
        print(f"{name:>13}", end="")
    print()
    for i, name in enumerate(VAR_NAMES):
        print(f"  {name:>14}", end="")
        for j in range(n_vars):
            val = A[i, j]
            print(f"{val:>13.4f}", end="")
        print()

print("\n\n3. PAIRWISE GRANGER CAUSALITY TESTS")
print("-" * 50)

def granger_test_from_var(data, var_result, cause_idx, effect_idx):
    """
    Test if variable cause_idx Granger-causes variable effect_idx
    using the estimated VAR coefficients.
    H0: all coefficients of cause variable in effect equation = 0
    """
    p = var_result["lag"]
    Y = data.values
    T, k = Y.shape

    Y_dep = Y[p:, effect_idx]
    n = len(Y_dep)

    # Unrestricted model (full VAR equation for effect variable)
    X_full = np.ones((n, 1))
    for lag in range(1, p + 1):
        X_full = np.column_stack([X_full, Y[p-lag:T-lag]])

    beta_full = np.linalg.lstsq(X_full, Y_dep, rcond=None)[0]
    resid_full = Y_dep - X_full @ beta_full
    ssr_full = np.sum(resid_full ** 2)

    # Restricted model (exclude cause variable lags)
    cols_to_keep = [0]  # constant
    for lag in range(1, p + 1):
        for j in range(k):
            if j != cause_idx:
                cols_to_keep.append(1 + (lag - 1) * k + j)

    X_restricted = X_full[:, cols_to_keep]
    beta_r = np.linalg.lstsq(X_restricted, Y_dep, rcond=None)[0]
    resid_r = Y_dep - X_restricted @ beta_r
    ssr_r = np.sum(resid_r ** 2)

    # F-test
    q = p  # number of restrictions
    k_full = X_full.shape[1]
    f_stat = ((ssr_r - ssr_full) / q) / (ssr_full / (n - k_full))
    p_value = 1 - stats.f.cdf(f_stat, q, n - k_full)

    return f_stat, p_value

# Compute all pairwise tests
print(f"\n  {'Cause':>14} → {'Effect':>14}   F-stat    p-value   Significant")
print("  " + "-" * 65)

gc_matrix = np.zeros((n_vars, n_vars))
gc_pvalues = np.zeros((n_vars, n_vars))

for i in range(n_vars):
    for j in range(n_vars):
        if i != j:
            f_stat, p_val = granger_test_from_var(df, var_result, i, j)
            gc_matrix[j, i] = f_stat  # j→i means: i-th equation, j-th cause
            gc_pvalues[j, i] = p_val
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            if p_val < 0.1:
                print(f"  {VAR_NAMES[i]:>14} → {VAR_NAMES[j]:>14}   "
                      f"{f_stat:>7.3f}   {p_val:>8.4f}   {sig}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: SPARSE NETWORK CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n4. SPARSE NETWORK CONSTRUCTION")
print("-" * 50)

# Method: Significance thresholding at 5% level
ALPHA = 0.05

# Compute edge weights: w_ij = Σ_k |a_ij^(k)|
weight_matrix = np.zeros((n_vars, n_vars))
for lag, A in var_result["coef_matrices"].items():
    weight_matrix += np.abs(A)

# Apply significance mask
edge_matrix = np.zeros((n_vars, n_vars))
for i in range(n_vars):
    for j in range(n_vars):
        if i != j and gc_pvalues[i, j] < ALPHA:
            edge_matrix[i, j] = weight_matrix[i, j]

print(f"\n  Significance threshold: α = {ALPHA}")
n_edges = np.sum(edge_matrix > 0)
n_possible = n_vars * (n_vars - 1)
print(f"  Edges retained: {int(n_edges)} / {n_possible} possible ({100*n_edges/n_possible:.0f}%)")

print(f"\n  Directed edges (j → i):")
for i in range(n_vars):
    for j in range(n_vars):
        if edge_matrix[i, j] > 0:
            # Get signed weight
            signed_weight = sum(var_result["coef_matrices"][lag][i, j]
                               for lag in var_result["coef_matrices"])
            direction = "+" if signed_weight > 0 else "-"
            print(f"    {VAR_NAMES[j]:>14} → {VAR_NAMES[i]:<14}  "
                  f"weight = {edge_matrix[i,j]:.4f} ({direction})")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: DIEBOLD-YILMAZ SPILLOVER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n5. DIEBOLD-YILMAZ SPILLOVER ANALYSIS")
print("-" * 50)

def compute_ma_coefficients(coef_matrices, k, H=20):
    """Compute MA representation coefficients Phi_0, ..., Phi_H"""
    p = max(coef_matrices.keys())
    Phi = [np.eye(k)]

    for h in range(1, H + 1):
        phi_h = np.zeros((k, k))
        for lag in range(1, min(h, p) + 1):
            phi_h += coef_matrices[lag] @ Phi[h - lag]
        Phi.append(phi_h)

    return Phi

def compute_generalised_fevd(Phi, Sigma, H=10):
    """
    Compute generalised FEVD (Diebold-Yilmaz approach).
    Does not depend on ordering (unlike Cholesky).
    """
    k = Sigma.shape[0]
    sigma_diag = np.diag(Sigma)

    # Numerator: for each (i,j), sum_h (e_i' Phi_h Sigma e_j)^2 / sigma_jj
    fevd = np.zeros((k, k))

    for i in range(k):
        denom = 0  # total forecast error variance of variable i
        for h in range(H + 1):
            denom += (Phi[h] @ Sigma @ Phi[h].T)[i, i]

        for j in range(k):
            numerator = 0
            for h in range(H + 1):
                numerator += (Phi[h] @ Sigma[:, j])[i] ** 2
            numerator /= sigma_diag[j]
            fevd[i, j] = numerator / denom if denom > 0 else 0

    # Normalise rows to sum to 1
    row_sums = fevd.sum(axis=1, keepdims=True)
    fevd = fevd / row_sums

    return fevd

# Compute
Phi = compute_ma_coefficients(var_result["coef_matrices"], n_vars, H=20)
fevd = compute_generalised_fevd(Phi, var_result["sigma"], H=10)

# Print FEVD table
print(f"\n  Spillover Table (H=10):")
print(f"  {'To / From':>14}", end="")
for name in VAR_NAMES:
    print(f"{name:>12}", end="")
print(f"{'FROM others':>14}")
print("  " + "-" * (14 + 12 * n_vars + 14))

directional_from = np.zeros(n_vars)
directional_to = np.zeros(n_vars)

for i in range(n_vars):
    print(f"  {VAR_NAMES[i]:>14}", end="")
    from_others = 0
    for j in range(n_vars):
        pct = fevd[i, j] * 100
        print(f"{pct:>11.1f}%", end="")
        if i != j:
            from_others += pct
    directional_from[i] = from_others
    print(f"{from_others:>13.1f}%")

print(f"  {'TO others':>14}", end="")
for j in range(n_vars):
    to_others = sum(fevd[i, j] * 100 for i in range(n_vars) if i != j)
    directional_to[j] = to_others
    print(f"{to_others:>11.1f}%", end="")
print()

# Net spillovers
print(f"\n  {'NET spillover':>14}", end="")
for j in range(n_vars):
    net = directional_to[j] - directional_from[j]
    print(f"{net:>+11.1f}%", end="")
print()

# Total spillover
total_spillover = np.sum(directional_from) / (np.sum(fevd) * 100) * 100
print(f"\n  Total Spillover Index: {total_spillover:.1f}%")

# Classify variables
print(f"\n  Variable Classification:")
for i in range(n_vars):
    net = directional_to[i] - directional_from[i]
    if net > 0:
        role = "NET TRANSMITTER ▶"
    else:
        role = "NET RECEIVER    ◀"
    print(f"    {VAR_NAMES[i]:>14}: Net = {net:>+7.1f}%  →  {role}")

print("\n\n6. NETWORK VISUALIZATION")
print("-" * 50)

def draw_network(edge_matrix, var_names, fevd, directional_to, directional_from,
                 filename="network_graph.png"):
    """Draw directed network with nodes and weighted edges"""
    k = len(var_names)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # ─── Plot 1: Granger Causality Network ───
    ax = axes[0]
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title("Granger Causality Network\n(Direct Predictive Effects)", fontsize=14, fontweight='bold')
    ax.axis('off')

    # Position nodes in a circle
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False) - np.pi/2
    positions = [(np.cos(a), np.sin(a)) for a in angles]

    # Net spillover for node coloring
    net_spill = directional_to - directional_from

    # Draw edges
    for i in range(k):
        for j in range(k):
            if edge_matrix[i, j] > 0:
                # Arrow from j to i
                x0, y0 = positions[j]
                x1, y1 = positions[i]

                # Shorten arrow to not overlap with nodes
                dx, dy = x1 - x0, y1 - y0
                dist = np.sqrt(dx**2 + dy**2)
                shrink = 0.18
                x0_s = x0 + shrink * dx / dist
                y0_s = y0 + shrink * dy / dist
                x1_s = x1 - shrink * dx / dist
                y1_s = y1 - shrink * dy / dist

                width = edge_matrix[i, j] * 3
                ax.annotate("", xy=(x1_s, y1_s), xytext=(x0_s, y0_s),
                           arrowprops=dict(arrowstyle="-|>", color="#555555",
                                          lw=max(0.5, width),
                                          mutation_scale=15, alpha=0.6))

    # Draw nodes
    for i, (x, y) in enumerate(positions):
        color = "#e74c3c" if net_spill[i] > 0 else "#3498db"
        size = 800 + abs(net_spill[i]) * 30
        ax.scatter(x, y, s=size, c=color, zorder=5, edgecolors='white', linewidth=2)
        ax.text(x, y - 0.25, var_names[i], ha='center', va='top', fontsize=10,
               fontweight='bold')

    # Legend
    red_patch = mpatches.Patch(color='#e74c3c', label='Net Transmitter')
    blue_patch = mpatches.Patch(color='#3498db', label='Net Receiver')
    ax.legend(handles=[red_patch, blue_patch], loc='lower left', fontsize=10)

    # ─── Plot 2: FEVD Heatmap ───
    ax = axes[1]
    im = ax.imshow(fevd * 100, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(var_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(var_names, fontsize=10)
    ax.set_xlabel("Shock Source (j)", fontsize=12)
    ax.set_ylabel("Affected Variable (i)", fontsize=12)
    ax.set_title("FEVD Spillover Matrix (%)\n(θ_ij: variance share of i due to j)", fontsize=14, fontweight='bold')

    # Add text annotations
    for i in range(k):
        for j in range(k):
            val = fevd[i, j] * 100
            color = "white" if val > 30 else "black"
            ax.text(j, i, f"{val:.1f}", ha='center', va='center',
                   fontsize=9, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, label="Variance Share (%)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Network visualization saved: {filename}")

draw_network(edge_matrix, VAR_NAMES, fevd, directional_to, directional_from)

print("\n\n7. ROLLING WINDOW SPILLOVER ANALYSIS")
print("-" * 50)

window_size = 80  # 80 months
step = 5          # step by 5 months

rolling_spillovers = []

for start in range(0, T - window_size, step):
    end = start + window_size
    window_data = df.iloc[start:end]

    try:
        # Estimate VAR on window
        w_var = estimate_var_full.__wrapped__(window_data, max_lag=3) \
            if hasattr(estimate_var_full, '__wrapped__') else None

        # Simple VAR estimation for rolling window
        Y = window_data.values
        n_w, k_w = Y.shape
        p_w = 2

        Y_dep = Y[p_w:]
        n_obs = len(Y_dep)

        X_w = np.ones((n_obs, 1))
        for lag in range(1, p_w + 1):
            X_w = np.column_stack([X_w, Y[p_w-lag:n_w-lag]])

        B_w = np.linalg.lstsq(X_w, Y_dep, rcond=None)[0]
        resid_w = Y_dep - X_w @ B_w
        Sigma_w = (resid_w.T @ resid_w) / n_obs

        coef_w = {}
        for lag in range(1, p_w + 1):
            start_idx = 1 + (lag - 1) * k_w
            coef_w[lag] = B_w[start_idx:start_idx+k_w, :].T

        Phi_w = compute_ma_coefficients(coef_w, k_w, H=10)
        fevd_w = compute_generalised_fevd(Phi_w, Sigma_w, H=10)

        total_spill = (np.sum(fevd_w) - np.trace(fevd_w)) / np.sum(fevd_w) * 100

        rolling_spillovers.append({
            "Period_Start": start,
            "Period_End": end,
            "Total_Spillover": total_spill,
            "Mid_Point": (start + end) / 2
        })

    except Exception as e:
        pass

if rolling_spillovers:
    roll_df = pd.DataFrame(rolling_spillovers)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(roll_df["Mid_Point"], roll_df["Total_Spillover"],
           color="#e74c3c", linewidth=2)
    ax.fill_between(roll_df["Mid_Point"], roll_df["Total_Spillover"],
                   alpha=0.15, color="#e74c3c")
    ax.axhline(y=roll_df["Total_Spillover"].mean(), color='gray',
              linestyle='--', alpha=0.7, label=f'Mean = {roll_df["Total_Spillover"].mean():.1f}%')
    ax.set_xlabel("Time Period (observation index)", fontsize=12)
    ax.set_ylabel("Total Spillover Index (%)", fontsize=12)
    ax.set_title("Rolling Window Total Spillover Index\n(Window = 80 months, Step = 5 months)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rolling_spillover_index.png"),
               dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Rolling spillover plot saved")
    print(f"  Mean total spillover: {roll_df['Total_Spillover'].mean():.1f}%")
    print(f"  Max total spillover: {roll_df['Total_Spillover'].max():.1f}%")
    print(f"  Min total spillover: {roll_df['Total_Spillover'].min():.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("COMPLETE ANALYSIS SUMMARY")
print("=" * 70)

print(f"""
Pipeline completed:
  ✓ Step 1: VAR({p}) estimated with AIC lag selection
  ✓ Step 2: Pairwise Granger causality tests (α = {ALPHA})
  ✓ Step 3: Sparse network constructed ({int(n_edges)} significant edges)
  ✓ Step 4: Edge weights computed (w_ij = Σ|a_ij^(k)|)
  ✓ Step 5: FEVD and Diebold-Yilmaz spillovers computed
  ✓ Step 6: Network visualised
  ✓ Step 7: Rolling window analysis

Key findings:
  - Total spillover index: {total_spillover:.1f}%
  - Net transmitters: {', '.join(VAR_NAMES[i] for i in range(n_vars) if directional_to[i] > directional_from[i])}
  - Net receivers: {', '.join(VAR_NAMES[i] for i in range(n_vars) if directional_to[i] <= directional_from[i])}

Output files (in your Downloads):
  - 5_Network_Spillover/network_graph.png
  - 5_Network_Spillover/rolling_spillover_index.png

To use with REAL DATA:
  Replace the simulated data (Section 1) with actual data from:
  - World Bank API (wbgapi) for inflation, unemployment, FX
  - Bloomberg for oil prices, Nifty, interest rates
  - MoSPI for Indian CPI/WPI
  Then re-run Sections 2-7 unchanged.
""")
