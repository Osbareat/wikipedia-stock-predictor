# Wikipedia Stock Predictor

A quantitative finance project exploring whether Wikipedia pageview data can predict stock returns.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This project tests the hypothesis Can the number of people checking a company’s Wikipedia page tell us anything about where its stock price is headed?


It’s the kind of question you ask when you’re genuinely trying to see the market from a different angle—using the same kind of “alternative data” hedge funds quietly love.

### What This Is Really About

At the heart of it, the project tests a simple idea:

If more people suddenly look up a company on Wikipedia, does that interest show up later in the stock price?

We treated Wikipedia pageviews as a potential signal—maybe weak, maybe strong—and built the entire pipeline to find out.

Companies tested: Amazon, Netflix, Walmart
Time period: Dec 2022 → Nov 2025 (3 years)
Tools: A full data-engineering + modeling workflow with time-series CV

---

## What We Found

Signal Strength

We engineered 33 different features from raw Wikipedia traffic.
Out of those, 7 features actually showed meaningful statistical significance (p < 0.05).

The strongest single feature?

The 30-day z-score, with an IC of 0.085
(Not huge, but definitely not noise.)

### Model Performance

**Regression (5-day return prediction):**
| Model | MAE | RMSE | R² | IC |
|-------|-----|------|----|----|
| Linear Regression | 0.0345 ± 0.012 | 0.0439 ± 0.015 | -0.368 ± 0.529 | +0.009 ± 0.365 |
| Ridge Regression | 0.0343 ± 0.012 | 0.0437 ± 0.015 | -0.348 ± 0.495 | +0.004 ± 0.361 |

**Classification (direction prediction):**
- In-sample accuracy: 59.8% (baseline: 58.7%)
- Cross-validation accuracy: 57.0% ± 11.9%
- Out-of-sample performance slightly below baseline

Not great R².

Classification: Up vs Down
	•	In-sample accuracy: 59.8% (baseline 58.7%)
	•	Cross-validation: 57% ± 11.9%
	•	Out-of-sample: slightly below baseline

So yes, there’s signal… but it’s faint.

Bottom Line

Wikipedia pageviews do contain a weak predictive element—but nowhere near enough to use alone.

Why?
	•	Markets digest public info fast
	•	Wikipedia interest often lags the event
	•	Alternative data shines when combined with fundamentals, technicals, and sentiment—not in isolation

---

## How Everything Was Built

Data Collection
	•	Wikipedia pageviews: via the Wikimedia REST API
	•	Stock data: pulled from Yahoo Finance (via yfinance)
	•	Features: 33 engineered metrics — rolling windows, z-scores, volatility, momentum, log transforms, you name it
	•	Validation: time-series cross-validation (14 expanding windows)

### Pipeline Architecture

```
Wikipedia API 
   → Fetching 
   → Feature Engineering 
   → Merge with Stock Data 
   → Model Training 
   → Time-Series Cross-Validation 
   → Evaluation
```

The notebook is organized into 6 modular sections:
Key Modules
	•	config.py — which companies we’re working on + API settings
	•	models.py — shared dataclasses/structures
	•	http_utils.py — resilient API client
	•	wpv_fetcher.py — Wikipedia data fetcher
	•	wpv_storage.py — save/load in CSV/Parquet/Excel
	•	wpv_analyzer.py — analytics helpers

Everything organized, modular, easy to reuse.

---

## Installation

# Clone the repo
git clone https://github.com/Osbareat/wikipedia-stock-predictor.git
cd wikipedia-stock-predictor

# Install dependencies
pip install -r requirements.txt

# Fire up the notebook
jupyter notebook full_analysis_pipeline.ipynb


### Requirements
Python 3.10+
pandas, numpy, scipy
scikit-learn, statsmodels
requests, yfinance
matplotlib, seaborn
pyarrow, openpyxl

---

## Quick Start Example
```
# Fetch Wikipedia pageviews
results = fetch_all_pageviews(
    companies=COMPANIES,
    start_date=date(2022, 12, 12),
    end_date=date(2025, 11, 28)
)

# Create features
df_features = generate_all_features(df_pageviews)

# Stock data
stock_data = fetch_stock_data(["AMZN", "NFLX", "WMT"])
stock_data = compute_returns(stock_data, horizons=[1, 5, 10, 20])

# Point-in-time merge
df_model = merge_data(df_features, stock_data)

# Compute information coefficients
ic_results = compute_information_coefficients(df_model, features, target='ret_5d')

# Cross-validate models
cv_results = time_series_cross_validate(df_model, features, target='ret_5d')
```

---

## Project Structure

```
wikipedia-stock-predictor/
├── full_analysis_pipeline.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── data/
    ├── wikipedia_pageviews/
    └── model_ready/
```

---

## Methodology

Feature Engineering

Built 33 features covering:
	•	Rolling windows: 7d, 14d, 30d, 90d
	•	Momentum: % changes, velocity, acceleration
	•	Statistical: z-scores, volatility, ranks
	•	Normalization: logs + standardization

Modeling Approach
	•	Information Coefficient analysis
	•	Linear/Logistic regression for interpretability
	•	Time-series cross-validation
	•	Multiple horizons: 1d, 5d, 10d, 20d

Evaluation Metrics
	•	MAE, RMSE, R²
	•	Information Coefficient
	•	Accuracy, Precision, Recall, F1, ROC-AUC
	•	Baseline comparisons

---

## Limitations & Future Work

### Current Limitations
- Small sample (3 companies, 3 years)
- Single data source (Wikipedia only)
- Simple models (linear/logistic regression)
- No transaction costs or portfolio construction

### Potential Extensions
- Expand to 20-50 companies across sectors
- Add fundamentals (P/E, revenue) and technicals (RSI, MACD)
- Test advanced models (Random Forest, XGBoost, LSTM)
- Integrate additional alternative data (Google Trends, news sentiment, social media)
- Implement portfolio optimization with risk management
- Build real-time inference system

---

## References

### Data Sources
- [Wikimedia REST API](https://wikimedia.org/api/rest_v1/) - Pageviews data
- [yfinance](https://pypi.org/project/yfinance/) - Stock prices (Yahoo Finance)

### Key Concepts
- **Alternative Data in Finance** - Kolanovic & Krishnamachari (2017), JPMorgan
- **Information Coefficient** - Grinold & Kahn (2000), Active Portfolio Management
- **Time-Series Cross-Validation** - Bergmeir & Benítez (2012)

---

## License

MIT License - see LICENSE file for details.

Data sourced from public APIs (Wikimedia, Yahoo Finance) in accordance with their terms of service.

**Disclaimer:** For educational purposes only. Not financial advice. Do not use for actual trading.

---

## Contact

**Ali Al Mazrouei**  
GitHub: [@Osbareat](https://github.com/Osbareat)  
Project: [github.com/Osbareat/wikipedia-stock-predictor](https://github.com/Osbareat/wikipedia-stock-predictor)

---

## Acknowledgments

Thanks to Wikimedia Foundation for API access, Yahoo Finance for historical data, and the open-source Python community.
