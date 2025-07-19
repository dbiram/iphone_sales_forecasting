# iPhone Sales Forecasting

[![Hugging Face Spaces](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-blue)](https://huggingface.co/spaces/dbiram/iphone-sales-forecast)

**Goal:** Forecast the next 8 weeks of weekly iPhone unit sales per model, incorporating:
- Historical sales
- Product lifecycle (cold start)
- Promotions / discounts
- Seasonality

This mirrors real-world retail forecasting use cases in inventory management and supply chain optimization.

---

## Project Overview

| Category                   | Details                                |
|----------------------------|----------------------------------------|
| Problem                    | Multi-product weekly sales forecasting |
| Data Source                | Proprietary iPhone sales data          |
| Forecast Horizon           | 8 weeks ahead                          |
| Key Metrics                | MAPE, WAPE, RMSE                       |
| Final Model                | LightGBM (gradient boosting)           |
| Alternatives Tested        | XGBoost, CatBoost                      |
| Deployment                 | FastAPI backend + Streamlit frontend   |

---

## Project Structure
```
iphone_sales_forecasting/
├── api/ # FastAPI backend (inference API)
├── app/ # Streamlit frontend (in French)
├── data/ # Raw data (excluded from GitHub)
├── models/ # Saved LightGBM model
├── notebooks/ # Analysis and modeling notebooks
├── src/ # Modular Python code
├── requirements.txt # Environment dependencies
└── README.md # Project documentation
```

---

## Methodology

### 1 EDA & Problem Framing
- Weekly sales analyzed per iPhone model
- Seasonality and lifecycle effects identified
- Promotions detected as sales spikes

### 2 Baseline Models
- Prophet and naive forecasting evaluated
- Volatile sales segments revealed baseline weaknesses

### 3 Supervised ML Models (Lag-Based)
- Features:
  - Lagged sales (`t-1`, `t-2`, `t-4`)
  - Rolling mean (4 weeks)
  - Promotions flag
  - Product age (weeks since launch)
  - Time encodings (week of year)
- Models evaluated:
  - LightGBM
  - XGBoost
  - CatBoost

### 4 Time-Based Backtesting
- Rolling-window evaluation (2019–2024)
- Forecasted next 8 weeks per window
- Evaluated using MAPE, WAPE, and RMSE

---

## Results

| Model      | Global WAPE | Global MAPE | Global RMSE |
|------------|------------|-------------|-------------|
| LightGBM   | 2.0 %      | ~909 %      | ~793 units  |
| XGBoost    | 1.4 %      | ~641 %      | ~909 units  |
| CatBoost   | 1.5 %      | ~1244 %     | ~712 units  |

> **Note:** High MAPE explained by small-actual weeks inflating % errors. WAPE preferred for business evaluation.

---

## Deployment Components

### FastAPI Backend
- `/predict/` endpoint (rolling forecast)
- Inputs:
  - Historical data (at least 4 weeks)
  - Future week commercial plan (prices, promotions, etc.)
- Output:
  - Sequential sales predictions

Run backend:

```bash
uvicorn api.main:app --reload
```
Access API docs at: http://127.0.0.1:8000/docs

### Streamlit Frontend (in French)
- Upload historical sales as CSV
- Input forecast window + future commercial plan
- Visualize sales forecasts in a date-based chart

Run backend:

```bash
streamlit run app/app.py
```

## Key Visualizations

**LightGBM (iPhone 14):**  

![LightGBM Forecast](figures/lightgbm_forecast_vs_actual.png)

**XGBoost (iPhone 14):** 

![XGBoost Forecast](figures/xgboost_forecast_vs_actual.png)

**CatBoost (iPhone 14):** 

![CatBoost Forecast](figures/catboost_forecast_vs_actual.png)

## Key Takeaways

- Built multi-product sales forecasting pipeline
- Integrated promotions and seasonality handling
- Applied LightGBM with lag features and rolling statistics
- Time-based validation and error analysis per product
- Fully deployed:
  - API backend (FastAPI)
  - Business-facing UI (Streamlit, in French)

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train final model (optional)
python -m src.final_model

# Run API
uvicorn api.main:app --reload

# Run Frontend
streamlit run app/app.py
```

## Live Demo (in French): 
[Accéder à l'application Streamlit sur Hugging Face](https://huggingface.co/spaces/dbiram/iphone-sales-forecast)
