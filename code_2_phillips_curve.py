import wbgapi as wb
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os

DOWNLOADS = os.path.join(os.path.expanduser("~"), "Downloads")
OUTPUT_DIR = os.path.join(DOWNLOADS, "2_Phillips_Curve")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f" All files will be saved to:\n   {OUTPUT_DIR}\n")

# Model Definitions 

def linear_model(x, a, b):
    """Standard linear Phillips Curve: inflation = a + b * unemployment"""
    return a + b * x

def hyperbolic_model(x, a, b):
    """Hyperbolic (1/x) Phillips Curve: inflation = a + b / unemployment"""
    return a + b / x

def compute_r_squared(y_actual, y_predicted):
    """Compute R² (coefficient of determination)"""
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - (ss_res / ss_tot)
print("=" * 70)
print("PHILLIPS CURVE ANALYSIS")
print("=" * 70)
print("\nStep 1: Identifying top 50 GDP countries...")

gdp_data = wb.data.DataFrame("NY.GDP.MKTP.CD", time=2022, labels=True)
gdp_data = gdp_data.reset_index()

# Clean and sort
if "YR2022" in gdp_data.columns:
    gdp_col = "YR2022"
elif 2022 in gdp_data.columns:
    gdp_col = 2022
else:
    # Try to find the GDP column
    year_cols = [c for c in gdp_data.columns if "2022" in str(c) or "2021" in str(c)]
    gdp_col = year_cols[0] if year_cols else gdp_data.columns[-1]

gdp_data = gdp_data.dropna(subset=[gdp_col])
gdp_data = gdp_data.sort_values(gdp_col, ascending=False)
top_50_codes = gdp_data["economy"].head(50).tolist()

print(f"  Top 50 economies identified: {top_50_codes[:10]}... (showing first 10)")
print("\nStep 2: Fetching inflation and unemployment data...")

TIME_RANGE = range(2000, 2024)
inflation_df = wb.data.DataFrame("FP.CPI.TOTL.ZG", economy=top_50_codes,
                                  time=TIME_RANGE, labels=True)
inflation_df = inflation_df.reset_index()
unemp_df = wb.data.DataFrame("SL.UEM.TOTL.ZS", economy=top_50_codes,
                              time=TIME_RANGE, labels=True)
unemp_df = unemp_df.reset_index()

print("  ✓ Data fetched")
def reshape_wb_data(df, value_name):
    """Convert wide WB format to long format"""
    year_cols = [c for c in df.columns if c.startswith("YR")]
    id_vars = ["economy"]
    if "Country" in df.columns:
        id_vars.append("Country")
    df_long = df.melt(id_vars=id_vars, value_vars=year_cols,
                       var_name="Year", value_name=value_name)
    df_long["Year"] = df_long["Year"].str.replace("YR", "").astype(int)
    return df_long

inf_long = reshape_wb_data(inflation_df, "Inflation")
unemp_long = reshape_wb_data(unemp_df, "Unemployment")

# Merge
merged = inf_long.merge(unemp_long[["economy", "Year", "Unemployment"]],
                         on=["economy", "Year"], how="inner")
merged = merged.dropna(subset=["Inflation", "Unemployment"])
merged = merged[(merged["Inflation"] > -10) & (merged["Inflation"] < 50)]
merged = merged[merged["Unemployment"] > 0.1]  # Avoid division by zero for 1/x

print(f"  ✓ Merged dataset: {len(merged)} observations, {merged['economy'].nunique()} countries")
print("\n" + "=" * 70)
print("TURKEY ANALYSIS (Detailed)")
print("=" * 70)

turkey = merged[merged["economy"] == "TUR"].copy()
if len(turkey) < 5:
    # Try alternative code
    turkey = merged[merged["economy"].str.contains("TUR|Turkey", case=False, na=False)].copy()

if len(turkey) >= 5:
    x_tur = turkey["Unemployment"].values
    y_tur = turkey["Inflation"].values

    # Linear fit
    try:
        popt_lin, _ = curve_fit(linear_model, x_tur, y_tur)
        y_pred_lin = linear_model(x_tur, *popt_lin)
        r2_lin = compute_r_squared(y_tur, y_pred_lin)
        print(f"\n  Linear fit:  inflation = {popt_lin[0]:.2f} + ({popt_lin[1]:.2f}) × unemployment")
        print(f"               R² = {r2_lin:.4f}")
    except:
        r2_lin = np.nan

    # Hyperbolic fit (1/x)
    try:
        popt_hyp, _ = curve_fit(hyperbolic_model, x_tur, y_tur, p0=[1, 1], maxfev=5000)
        y_pred_hyp = hyperbolic_model(x_tur, *popt_hyp)
        r2_hyp = compute_r_squared(y_tur, y_pred_hyp)
        print(f"\n  Hyperbolic:  inflation = {popt_hyp[0]:.2f} + ({popt_hyp[1]:.2f}) / unemployment")
        print(f"               R² = {r2_hyp:.4f}")
    except:
        r2_hyp = np.nan

    # Plot Turkey
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter + linear
    x_range = np.linspace(x_tur.min(), x_tur.max(), 100)
    axes[0].scatter(x_tur, y_tur, color='#e74c3c', s=60, alpha=0.7, edgecolors='white')
    if not np.isnan(r2_lin):
        axes[0].plot(x_range, linear_model(x_range, *popt_lin), 'b-', linewidth=2,
                     label=f'Linear: R² = {r2_lin:.4f}')
    axes[0].set_xlabel("Unemployment (%)", fontsize=12)
    axes[0].set_ylabel("Inflation (%)", fontsize=12)
    axes[0].set_title("Turkey: Linear Phillips Curve", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Scatter + hyperbolic
    axes[1].scatter(x_tur, y_tur, color='#e74c3c', s=60, alpha=0.7, edgecolors='white')
    if not np.isnan(r2_hyp):
        axes[1].plot(x_range, hyperbolic_model(x_range, *popt_hyp), 'g-', linewidth=2,
                     label=f'Hyperbolic (1/x): R² = {r2_hyp:.4f}')
    axes[1].set_xlabel("Unemployment (%)", fontsize=12)
    axes[1].set_ylabel("Inflation (%)", fontsize=12)
    axes[1].set_title("Turkey: Hyperbolic Phillips Curve", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "turkey_phillips_curve.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  ✓ Turkey plot saved")
else:
    print("  ⚠ Not enough Turkey data")

print("\n" + "=" * 70)
print("TOP 50 GDP COUNTRIES - PHILLIPS CURVE FITS")
print("=" * 70)

results = []

for country_code in top_50_codes:
    country_data = merged[merged["economy"] == country_code].copy()

    if len(country_data) < 5:
        continue

    x = country_data["Unemployment"].values
    y = country_data["Inflation"].values

    country_name = country_code
    if "Country" in country_data.columns:
        names = country_data["Country"].dropna().unique()
        if len(names) > 0:
            country_name = names[0]

    # Linear fit
    r2_lin = np.nan
    slope_lin = np.nan
    try:
        popt, _ = curve_fit(linear_model, x, y)
        y_pred = linear_model(x, *popt)
        r2_lin = compute_r_squared(y, y_pred)
        slope_lin = popt[1]
    except:
        pass

    # Hyperbolic fit
    r2_hyp = np.nan
    coeff_hyp = np.nan
    try:
        popt, _ = curve_fit(hyperbolic_model, x, y, p0=[1, 1], maxfev=5000)
        y_pred = hyperbolic_model(x, *popt)
        r2_hyp = compute_r_squared(y, y_pred)
        coeff_hyp = popt[1]
    except:
        pass

    results.append({
        "Country_Code": country_code,
        "Country": country_name,
        "N_obs": len(country_data),
        "R2_Linear": r2_lin,
        "Slope_Linear": slope_lin,
        "R2_Hyperbolic": r2_hyp,
        "Coeff_Hyperbolic": coeff_hyp,
        "Better_Fit": "Hyperbolic" if (not np.isnan(r2_hyp) and r2_hyp > r2_lin) else "Linear"
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("R2_Linear", ascending=False)

# Display results
print(f"\n{'Country':<25} {'N':>3} {'R² Linear':>10} {'R² Hyper':>10} {'Better Fit':>12}")
print("-" * 65)
for _, row in results_df.iterrows():
    print(f"{str(row['Country'])[:24]:<25} {row['N_obs']:>3} "
          f"{row['R2_Linear']:>10.4f} {row['R2_Hyperbolic']:>10.4f} {row['Better_Fit']:>12}")

# Save results
results_df.to_csv(os.path.join(OUTPUT_DIR, "phillips_curve_results_top50.csv"), index=False)

# Summary Statistics
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

valid = results_df.dropna(subset=["R2_Linear", "R2_Hyperbolic"])
n_hyp_better = (valid["Better_Fit"] == "Hyperbolic").sum()
n_lin_better = (valid["Better_Fit"] == "Linear").sum()

print(f"\n  Total countries analyzed: {len(results_df)}")
print(f"  Hyperbolic (1/x) fit better: {n_hyp_better} countries")
print(f"  Linear fit better: {n_lin_better} countries")
print(f"\n  Average R² (Linear): {valid['R2_Linear'].mean():.4f}")
print(f"  Average R² (Hyperbolic): {valid['R2_Hyperbolic'].mean():.4f}")
print(f"\n  Negative slopes (expected for Phillips Curve):")
print(f"    Linear: {(valid['Slope_Linear'] < 0).sum()} / {len(valid)} countries")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# R² comparison bar chart (top 15)
top15 = results_df.head(15).copy()
x_pos = np.arange(len(top15))
width = 0.35

axes[0].barh(x_pos - width/2, top15["R2_Linear"], width, label="Linear", color="#3498db", alpha=0.8)
axes[0].barh(x_pos + width/2, top15["R2_Hyperbolic"], width, label="Hyperbolic (1/x)", color="#e74c3c", alpha=0.8)
axes[0].set_yticks(x_pos)
axes[0].set_yticklabels(top15["Country_Code"], fontsize=9)
axes[0].set_xlabel("R²")
axes[0].set_title("Phillips Curve R² Comparison\n(Top 15 by Linear R²)", fontsize=12)
axes[0].legend()
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

# Scatter: Linear R² vs Hyperbolic R²
axes[1].scatter(valid["R2_Linear"], valid["R2_Hyperbolic"],
               s=50, alpha=0.7, color="#2ecc71", edgecolors="white")
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Equal fit line")
axes[1].set_xlabel("R² (Linear)", fontsize=12)
axes[1].set_ylabel("R² (Hyperbolic 1/x)", fontsize=12)
axes[1].set_title("Linear vs Hyperbolic Fit\n(Above line = hyperbolic better)", fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "phillips_curve_comparison_top50.png"), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✓ All outputs saved!")
print(f"\n FILES IN YOUR DOWNLOADS: 2_Phillips_Curve/")
print("-" * 60)
for f in sorted(os.listdir(OUTPUT_DIR)):
    full_path = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(full_path) / 1024
    print(f"  {f:<50s} ({size_kb:.0f} KB)")
print("-" * 60)
