# Macroeconomic Analysis & Diebold Yilmaz Spillover Project

A collection of Python scripts for macroeconomic data analysis, covering everything from data extraction to advanced network spillover modeling using the Diebold-Yilmaz framework.

## Project Structure
## Scripts Overview

### Code 1 — World Bank API Data Extraction
Connects to the World Bank database using `wbgapi` and downloads 10 macroeconomic indicators (CPI inflation, unemployment, FDI, government debt, exchange rate, GDP, external debt, WPI, real interest rate) for all countries from 2000–2023. Outputs individual CSVs per indicator and a unified panel dataset.

### Code 2 — Phillips Curve Analysis
Tests the Phillips Curve (inverse relationship between inflation and unemployment) for the top 50 countries by GDP. Fits two models for each country:
- **Linear**: `inflation = a + b × unemployment`
- **Hyperbolic**: `inflation = a + b / unemployment`

Compares R² values across models and countries. Includes a detailed case study for Turkey.

### Code 3 — Mandi/MoSPI Data Quality Assessment
Pulls daily agricultural commodity prices from India's data.gov.in API. Performs a full data quality audit — missing values, duplicates, price outliers, formatting issues — to determine if the data is usable without heavy cleaning.

### Code 4 — FX Rate vs Inflation Pass-through (India)
Analyzes how much the USD/INR exchange rate drives India's inflation:
- Lagged cross-correlations (0–5 year lags)
- Granger causality tests (does FX statistically predict inflation?)
- VAR model + FEVD (quantifies how much of inflation variance is explained by FX)

### Code 5 — Network Spillover Analysis (Diebold-Yilmaz)
Implements the full Diebold-Yilmaz (2012) spillover framework:
- Variables: FX rate, oil price, CPI, WPI, interest rate, Nifty
- VAR estimation → Generalized FEVD → Spillover table
- Network graph showing direction and strength of shock transmission
- Rolling window analysis to track how connectedness evolves over time

Key result: FX rate and oil price are **net transmitters** of shocks; CPI, WPI, interest rate, and Nifty are **net receivers**.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python code_1_worldbank_api.py
python code_2_phillips_curve.py
python code_3_mandi_mospi_data.py
python code_4_fx_inflation_passthrough.py
python code_5_network_spillover_analysis.py
```

## References

- Diebold, F.X. & Yilmaz, K. (2012). *Better to Give than to Receive: Predictive Directional Measurement of Volatility Spillovers.* International Journal of Forecasting.
- Phillips, A.W. (1958). *The Relation Between Unemployment and the Rate of Change of Money Wage Rates in the United Kingdom, 1861–1957.*
- World Bank Open Data: https://data.worldbank.org
