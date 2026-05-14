
import wbgapi as wb
import pandas as pd
import os

# ─── Output directory ────────────────────────────────────────────────────────
# All CSV files will be saved to your Desktop inside Project_Output folder
DESKTOP = os.path.join(os.path.expanduser("~"), "Downloads")
OUTPUT_DIR = os.path.join(DESKTOP, "1_WorldBank_Data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"📂 All files will be saved to:\n   {OUTPUT_DIR}\n")

# ─── Define indicators ───────────────────────────────────────────────────────
# These are the standard World Bank indicator codes
INDICATORS = {
    "FP.CPI.TOTL.ZG": "Inflation_CPI_annual_pct",
    "SL.UEM.TOTL.ZS": "Unemployment_pct_total_labor",
    "BX.KLT.DINV.WD.GD.ZS": "FDI_net_inflows_pct_GDP",
    "GC.DOD.TOTL.GD.ZS": "Central_govt_debt_pct_GDP",
    "PA.NUS.FCRF": "Official_exchange_rate_LCU_per_USD",
    "NY.GDP.MKTP.CD": "GDP_current_USD",
    "NY.GDP.MKTP.KD.ZG": "GDP_growth_annual_pct",
    "DT.DOD.DECT.CD": "External_debt_stocks_current_USD",
    "FP.WPI.TOTL": "Wholesale_price_index",
    "FR.INR.RINR": "Real_interest_rate_pct",
}

TIME_RANGE = range(2000, 2024)  # 2000 to 2023

print("=" * 70)
print("WORLD BANK DATA EXTRACTION")
print("=" * 70)

# ─── Extract each indicator ──────────────────────────────────────────────────
all_data = {}

for code, name in INDICATORS.items():
    print(f"\nFetching: {name} ({code})...")
    try:
        df = wb.data.DataFrame(code, time=TIME_RANGE, labels=True)
        df = df.reset_index()

        # The DataFrame has country codes as index and years as columns
        # Melt it into long format for easier analysis
        if "economy" in df.columns:
            id_vars = ["economy"]
            if "Country" in df.columns:
                id_vars.append("Country")

            # Year columns are like "YR2000", "YR2001", etc.
            year_cols = [c for c in df.columns if c.startswith("YR")]
            df_long = df.melt(
                id_vars=id_vars,
                value_vars=year_cols,
                var_name="Year",
                value_name=name
            )
            df_long["Year"] = df_long["Year"].str.replace("YR", "").astype(int)
            all_data[name] = df_long
            print(f"  ✓ Got {len(df_long)} observations across {df_long['economy'].nunique()} countries")
        else:
            print(f"  ⚠ Unexpected format, saving raw")
            all_data[name] = df

    except Exception as e:
        print(f"  ✗ Error: {e}")

# ─── Save individual CSVs ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SAVING DATA FILES")
print("=" * 70)

for name, df in all_data.items():
    filepath = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")

# ─── Create a merged panel dataset ───────────────────────────────────────────
print("\nMerging into unified panel dataset...")
merged = None
for name, df in all_data.items():
    if "economy" in df.columns and "Year" in df.columns:
        subset = df[["economy", "Year", name]].copy()
        if merged is None:
            merged = subset
        else:
            merged = merged.merge(subset, on=["economy", "Year"], how="outer")

if merged is not None:
    merged = merged.sort_values(["economy", "Year"]).reset_index(drop=True)
    merged.to_csv(os.path.join(OUTPUT_DIR, "unified_panel_dataset.csv"), index=False)
    print(f"  ✓ Unified panel: {merged.shape[0]} rows × {merged.shape[1]} columns")
    print(f"  Countries: {merged['economy'].nunique()}")
    print(f"  Year range: {merged['Year'].min()} - {merged['Year'].max()}")

# ─── Quick summary statistics ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY STATISTICS (Latest available year per indicator)")
print("=" * 70)

for name in INDICATORS.values():
    if name in merged.columns:
        latest = merged.dropna(subset=[name]).groupby("economy")[name].last()
        print(f"\n{name}:")
        print(f"  Mean: {latest.mean():.2f}, Median: {latest.median():.2f}")
        print(f"  Min: {latest.min():.2f}, Max: {latest.max():.2f}")
        print(f"  Std: {latest.std():.2f}")

print("\n✓ All data extraction complete!")
print(f"\n📂 FILES SAVED TO: {OUTPUT_DIR}")
print("-" * 60)

# List every file that was actually created
for f in sorted(os.listdir(OUTPUT_DIR)):
    full_path = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(full_path) / 1024
    print(f"  {f:<50s} ({size_kb:.0f} KB)")

print("-" * 60)
print(f"\nTo open this folder:")
print(f"  Windows:  explorer \"{OUTPUT_DIR}\"")
print(f"  Mac:      open \"{OUTPUT_DIR}\"")
print(f"  Linux:    xdg-open \"{OUTPUT_DIR}\"")
print(f"\nTo change save location, edit OUTPUT_DIR at the top of this script.")
