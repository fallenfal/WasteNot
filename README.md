# WasteNot: Proactive Demand Forecasting to Reduce Hospitality Food Waste
### A Comparative Analysis of Time-Series Algorithm Using an Exogenous-Aware ML Pipeline for SKU-Level Kitchen Optimization

Predicting daily and hourly sales for 64 menu items using multiple ML models, with a production pipeline that generates kitchen prep and ordering reports.

## What This Project Does

This is a demand forecasting system built for a breakfast/brunch café in Aberdeen. The idea is simple — if we can predict how many Bacon Baps or Big Breakfasts we'll sell tomorrow (or next week), the kitchen can prep the right amount of ingredients, reduce waste, and avoid running out of stuff mid-service.

The system forecasts sales for 64 specific menu items across multiple time horizons (1-day, 7-day, 30-day), then converts those predictions into ingredient-level quantities using a recipe database. So instead of just saying "you'll sell 12 Big Breakfasts", it tells the kitchen "you need 24 rashers of Back Bacon, 24 Fried Eggs, 36 Hash Browns..." etc.

## Models

I tried a bunch of different approaches to see which one actually works best for this kind of data. Each model has its own notebook where I train it, evaluate it on a blind November 2025 test set, and save the results to a shared SQLite database so I can compare them fairly.

### XGBoost (4 variants)
- **Simple Daily** (`simple_xgboost_daily.ipynb`) — basic lag features + weather/holidays, one global model for all products
- **Improved Daily** (`improved_xgboost_daily.ipynb`) — adds holiday proximity features, better weather aggregation (mean for temp, sum for precipitation, max for gusts), events data from Aberdeen
- **Simple Hourly** (`simple_xgboost_hourly.ipynb`) — same idea but at the hourly level (9 business hours per day, 8am-5pm)
- **Improved Hourly** (`improved_xgboost_hourly.ipynb`) — hourly version with the full feature set

All XGBoost models use Optuna for hyperparameter tuning (30 trials) and recursive forecasting — meaning when predicting day 2, the lag features use day 1's *prediction*, not the actual value. This avoids data leakage which was something I had to be really careful about.

### Prophet (2 variants)
- **Daily** (`prophet_daily.ipynb`) — Facebook Prophet with weather, holidays, and events as external regressors
- **Hourly** (`prophet_hourly.ipynb`) — same but at the hourly grain

Both Prophet notebooks use demand clustering — I group products by their sales behaviour (volume, volatility, weekend ratio, weather sensitivity, zero-fraction) using KMeans, then tune Prophet's hyperparameters separately for each cluster. The idea is that a high-volume steady seller like Bacon needs different changepoint/seasonality settings than a rare item like Vegan Sausage.

### LSTM (2 variants)
- **Global Daily** (`lstm_global_daily.ipynb`) — one LSTM trained on ALL 64 products simultaneously
- **Global Hourly** (`lstm_global_hourly.ipynb`) — same idea but hourly

The LSTM was the hardest one to get working. I had to run it through WSL on Windows to use the GPU because TensorFlow doesnt play nice with GPU on native Windows. There were a lot of memory issues — the hourly model especially, since 64 products × ~1400 days × 9 hours × 30-step sequences is a massive amount of data. I ended up using numpy memmap files to stream sequences from disk instead of holding everything in RAM, and had to downcast float64 to float32 to keep memory under control.
The logic for the use of memmap files is that the hourly model uses huge amount of RAM.

The LSTM also uses the same clustering approach as Prophet for Optuna tuning — run trials on a "hero product" (highest volume in each cluster) instead of all 64, which makes tuning actually feasible.

### ARIMA
- **Daily** (`arima.ipynb`) — ARIMA(1,1,1) using StatsForecast/Nixtla, no external features

This was mainly included as a baseline. ARIMA is univariate so it cant use weather or events, which limits it quite a bit for this use case.

## Evaluation

All models are evaluated on the same blind test period: **November 2-30, 2025**. Training data goes up to November 1st.

The main metrics I focused on:

- **WAPE** (Weighted Absolute Percentage Error) — this is the "north star" metric. It tells you the total absolute error as a percentage of total actual sales, weighted by volume. So high-sellers like Bacon Bap matter more than items that sell 1-2 per day. This is the metric the pipeline uses to auto-select the best model.
- **MASE** (Mean Absolute Scaled Error) — compares the model against a naive lag-1 baseline (just predicting yesterday's sales). MASE < 1 means the model is doing better than the naive approach.
- **MAE** — plain mean absolute error in units, useful for kitchen staff ("we're off by about 2 portions on average")
- **Bias** — whether the model tends to over-predict or under-predict (positive = over-prep, negative = under-prep)

Every notebook saves its results to `results/model_tracking.db` using the same SQLite schema.
The `model_comparison.ipynb` notebook compares all models using plots and tables.
For a visual representation of the results, `model_comparison.ipynb` must be run, also saves the results to 3 files, `results/comparison_*.csv`.

## Production Pipeline

`forecast_pipeline.py` is the deliverables of this project. It's a Python script that runs the all models depending what model is currently best according to WAPE.
`forecast_pipeline.py` also generates PDF reports for the kitchen (prep lists) and management (performance summaries).
`forecast_pipeline.py` has 2 flag implemented, `--november` and `--model`. This helps to run the pipeline on a specific model or a specific test set.
`forecast_pipeline.py` is the bit that ties everything together. It's designed to be run from the command line:

```bash
# Default: uses the best model (by WAPE) from the tracking DB
python forecast_pipeline.py

# Force a specific model
python forecast_pipeline.py --model xgb_improved_daily

# Run the November evaluation (same train/test split as the notebooks)
python forecast_pipeline.py --november
```

It does:
1. Queries the SQLite DB to find which model performed best
2. Retrains that model on latest data
3. Generates forecasts for 1-day, 7-day, and 30-day horizons
4. Converts product forecasts into ingredient quantities using the recipe database
5. Generates PDF reports for the kitchen (prep lists) and management (performance summaries)

The pipeline supports all model variants and can switch between them based on which one is currently performing best.

## Data Sources

All preprocessed CSVs live in `preprocesing_data/processed_csv/`:

- **Sales data** — hourly POS transaction data from Jan 2022 to Dec 2025. Each row is one hour, columns are product quantities sold.
- **Weather data** — hourly weather from Open-Meteo for Aberdeen (temperature, precipitation, wind, cloud cover, etc). I convert weather codes into binary flags: is_clear, is_cloudy, is_rain, is_snow.
- **Holidays** — UK bank holidays and Scottish school holidays with importance scores and days-until-next-holiday feature
- **Events** — Aberdeen local events (festivals, football matches) scraped and compiled into a timeline with importance flags

## Feature Engineering

The features vary a bit by model but the main ones across the board are:

- **Lag features**: sales 1, 7, and 30 days/hours ago (always recomputed recursively during forecasting to avoid leakage)
- **Rolling stats**: 7-day rolling mean and std, plus day-over-day difference
- **Time features**: cyclical encoding (sin/cos) for day-of-week, month, and hour-of-day. Also binary is_weekend flag.
- **Weather**: apparent temperature, precipitation, wind speed/gusts, visibility, humidity, cloud cover, plus the weather type flags
- **Holidays & events**: binary flags with importance scores, proximity features (is_holiday_lag_1, is_holiday_lead_1)

## Project Structure

```
├── arima_forecast/
│    └── arima.ipynb
├── xgboost_forecast/
│    └── simple_xgboost_daily.ipynb
│    └── simple_xgboost_hourly.ipynb
│    └── improved_xgboost_daily.ipynb
│    └── improved_xgboost_hourly.ipynb
├── prophet_forecast/
│    └── prophet_daily.ipynb
│    └── prophet_hourly.ipynb
├── lstm_forecast/
│    └── lstm_global_daily.ipynb
│    └── lstm_global_hourly.ipynb
├── menu/
│   └──  menu_items_ingredients.py
│    └──  Cookbook.pdf
├── pipeline/
│   └── forecast_pipeline.py
├── preprocesing_data/
│    └── eds_outputs/
│    └── eda_outputs.ipynb
│    └── events_data_processing.ipynb
│    └── holidays_data_processing.ipynb
│   └── sales_data_preprocessed.ipynb
│    └── weather_data_processing.ipynb
│   └── processed_csv/
│       ├── sales_data_preprocessed.csv
│       ├── weather_data_hourly.csv
│       ├── holidays_data_preprocessed.csv
│       └── aberdeen_events_master_timeline.csv
├── results/
│   └── model_tracking.db
│    └── CSV/PDF reports
└── model_comparison.ipynb
```

## Requirements

- Python 3.9+
- pandas, numpy, scikit-learn
- xgboost
- prophet (fbprophet)
- tensorflow (GPU recommended for LSTM — I used WSL + CUDA)
- statsforecast (for ARIMA)
- optuna
- reportlab (for PDF report generation)
- sqlite3 (standard library)

## Known Issues / Things I'd Improve

- The recipe database is hardcoded in Python — ideally this should be in a CSV or database table that kitchen staff can update themselves without touching code
- Some products have very low daily volume (1-2 units) which makes them really hard to forecast accurately for any model. The WAPE on these can be quite high even if the absolute error is tiny
- The hourly LSTM takes a very long time to train and needs careful memory management. On my setup (RTX GPU through WSL) it still took a while
- The November test set is only 29 days which is fine for daily models but a bit short for proper statistical significance on per-product metrics
- I haven't implemented automatic retraining scheduling yet — currently you run the pipeline manually
