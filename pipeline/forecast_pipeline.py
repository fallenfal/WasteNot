# ── Forecast pipeline ──
# Handles model retraining, forecasting (1, 7, 30 days), and report generation.
# Generates reports for kitchen prep, ordering, and stakeholders.
#
# Defaults to the best performing model from model_tracking.db (based on WAPE).
# The -november flag enables standardized evaluation for November 2025.

import os
import sys
import argparse
import sqlite3
import logging
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ── paths ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, 'preprocesing_data', 'processed_csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DB_PATH     = os.path.join(RESULTS_DIR, 'model_tracking.db')

# Optuna trials for retraining
OPTUNA_TRIALS = 30

# Operational hours: 8 AM to 5 PM (9 hours)
BUSINESS_HOURS = list(range(8, 17))
HOURS_PER_DAY  = 9

# Model Selection Strategy:
# Set to None for auto-selection based on WAPE.
# Manual overrides available for specific model testing.
FORCE_MODEL = 'xgb_improved_daily'

# grab the products list from the existing menu_items_ingredients.py
# instead of duplicating it here
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from menu_items_ingredients import recipes as _recipes_df
    PRODUCTS_TO_FORECAST = _recipes_df['product'].unique().tolist()
except ImportError:
    # fallback if the file isn't in the same directory
    PRODUCTS_TO_FORECAST = [
        'Avo & Hal Muffin','Avo, Egg & Bacon','Avo, Feta & Tom','Avocado on Toast',
        'Bacon','Bacon Bap','Bacon Egg Brioch','Bacon Waffle','Baked Beans',
        'Baked Beans JP','Bean Soldiers','Big Breakfast','Black Pudding',
        'Breakfast Hash','Breakfast Muffin','Breakfast Wrap','Buttd Mushrooms',
        'Cheese & Bean JP','Cheese JP','Chick Flatbread','Chicken Club',
        'Chilli Carne JP','Egg Bacon Brioch','Egg Bap','Extra Beans',
        'F.Eggs on Toast','Festive Stack','Fried Egg','Hash Brown',
        'Hash Brown Bites','Little Avo Toast','Little Bean Toas','Little Egg Toast',
        'Ltle Bfast Bacon','Ltle Bfast Saus','Mini Hash Browns','P.Eggs on Toast',
        'Poached Egg','Posh Beans','Roll & Butter','S.Eggs on Toast',
        'Sausage','Sausage Bap','Scrambled Egg','Streaky Bacon','Tattie Scone',
        'The Breakfast','Toasted Teacake','Tuna JP','Tuna Mayo Mix',
        'Tuna Melt Panini','Tuna Panini','Tuna Toastie','Veg Sausage Bap',
        'Vegan Breakfast','Vegan Sausage','Veggie Bap','Veggie Breakfast',
        'Bakery','White Toast Bread','Brown Toast Bread','Porridge',
        'Sourdough Toast Bread','Multiseed Toast Bread',
    ]

# maps what's stored in the DB to which runner function to call
MODEL_TYPE_TO_RUNNER = {
    'XGBoost_Simple_Daily':      'xgb_simple_daily',
    'XGBoost_Improved_Daily':    'xgb_improved_daily',
    'XGBoost_Simple_Hourly':     'xgb_simple_hourly',
    'XGBoost_Improved_Hourly':   'xgb_improved_hourly',
    'ARIMA':                     'arima',
    'Prophet_Daily':             'prophet_daily',
    'Prophet_Daily_Clustered':   'prophet_daily',
    'Prophet_Hourly':            'prophet_hourly',
    'Prophet_Hourly_Clustered':  'prophet_hourly',
    'LSTM_Global_Daily':         'lstm_daily',
    'LSTM_Daily':                'lstm_daily',
    'LSTM_Global_Hourly':        'lstm_hourly',
    'LSTM_Hourly':               'lstm_hourly',
}


# Not implemented yet, eventually move this to a CSV or DB table so the kitchen can update it
def get_recipes():
    try:
        from menu_items_ingredients import recipes
        return recipes
    except ImportError:
        pass

    # hardcoded fallback — copied from menu_items_ingredients.py
    data = [
        ("Avo & Hal Muffin","English Muffin",1,"pcs"),("Avo & Hal Muffin","Halloumi",2,"slices"),
        ("Avo & Hal Muffin","Avocado",0.5,"pcs"),("Avo & Hal Muffin","Lime",0.25,"pcs"),
        ("Avo & Hal Muffin","Sweet Chilli Jam",15,"g"),
        ("Avo, Egg & Bacon","Sourdough Loaf",1,"slices"),("Avo, Egg & Bacon","Avocado",0.5,"pcs"),
        ("Avo, Egg & Bacon","Lime",0.25,"pcs"),("Avo, Egg & Bacon","Unsmoked Streaky Bacon",2,"rashers"),
        ("Avo, Egg & Bacon","Fried Egg",1,"pcs"),("Avo, Egg & Bacon","Crushed Chillies",1,"g"),
        ("Avo, Egg & Bacon","Wild Rocket",5,"g"),
        ("Avo, Feta & Tom","Sourdough Loaf",1,"slices"),("Avo, Feta & Tom","Avocado",0.5,"pcs"),
        ("Avo, Feta & Tom","Lime",0.25,"pcs"),("Avo, Feta & Tom","Sun Dried Tomatoes",30,"g"),
        ("Avo, Feta & Tom","Feta Cheese",25,"g"),("Avo, Feta & Tom","Vinegar",10,"g"),
        ("Avocado on Toast","Sourdough Loaf",1,"slices"),("Avocado on Toast","Avocado",0.5,"pcs"),
        ("Avocado on Toast","Tomato",0.5,"pcs"),("Avocado on Toast","Crushed Chillies",1,"g"),
        ("Bacon","Back Bacon",1,"rashers"),
        ("Bacon Bap","Back Bacon",2,"rashers"),("Bacon Bap","Large White Bap",1,"pcs"),
        ("Bacon Bap","Unsalted Butter",10,"g"),
        ("Bacon Egg Brioch","Brioche Bun",1,"pcs"),("Bacon Egg Brioch","Fried Egg",1,"pcs"),
        ("Bacon Egg Brioch","Unsmoked Streaky Bacon",2,"rashers"),("Bacon Egg Brioch","Wild Rocket",5,"g"),
        ("Bacon Egg Brioch","Sweet Chilli Jam",10,"g"),("Bacon Egg Brioch","Coriander",2,"g"),
        ("Bacon Egg Brioch","Soft Cheese",35,"g"),("Bacon Egg Brioch","Crushed Chillies",0.5,"g"),
        ("Bacon Waffle","Belgian Sugar Waffles",2,"pcs"),("Bacon Waffle","Unsmoked Streaky Bacon",3,"rashers"),
        ("Bacon Waffle","Maple Syrup",10,"g"),
        ("Baked Beans","Baked Beans",80,"g"),
        ("Baked Beans JP","Baked Beans",80,"g"),("Baked Beans JP","Baking Potato",1,"pcs"),
        ("Baked Beans JP","Babyleaf Salad",15,"g"),("Baked Beans JP","Cherry Tomatoes",1,"pcs"),
        ("Bean Soldiers","50/50 Whole Bread",1,"slices"),("Bean Soldiers","Baked Beans",80,"g"),
        ("Big Breakfast","Baked Beans",80,"g"),("Big Breakfast","Pork Sausage",2,"pcs"),
        ("Big Breakfast","Back Bacon",2,"rashers"),("Big Breakfast","Fried Egg",2,"pcs"),
        ("Big Breakfast","Hash Brown",3,"pcs"),("Big Breakfast","Tattie Scone",1,"pcs"),
        ("Black Pudding","Black Pudding",1,"pcs"),
        ("Breakfast Hash","Hash Brown Bites",150,"g"),("Breakfast Hash","Red Pepper",0.5,"pcs"),
        ("Breakfast Hash","Spring Onion",10,"g"),("Breakfast Hash","Fried Egg",1,"pcs"),
        ("Breakfast Hash","Mixed Baked Beans",0.5,"tin"),("Breakfast Hash","Smoked Paprika",2,"g"),
        ("Breakfast Muffin","English Muffin",1,"pcs"),("Breakfast Muffin","Pork Sausage",1,"pcs"),
        ("Breakfast Muffin","Fried Egg",1,"pcs"),("Breakfast Muffin","Mild Cheese",1,"slices"),
        ("Breakfast Wrap","Tortilla Wrap",1,"slices"),("Breakfast Wrap","Tomato Chutney",15,"g"),
        ("Breakfast Wrap","Scrambled Egg",1,"pcs"),("Breakfast Wrap","Hash Brown",1,"pcs"),
        ("Breakfast Wrap","Pork Sausage",1,"pcs"),("Breakfast Wrap","Back Bacon",1,"rashers"),
        ("Buttd Mushrooms","Sourdough Loaf",1,"slices"),("Buttd Mushrooms","Unsalted Butter",10,"g"),
        ("Buttd Mushrooms","Fried Egg",1,"pcs"),("Buttd Mushrooms","Wild Rocket",5,"g"),
        ("Buttd Mushrooms","Sliced Mushroom",125,"g"),("Buttd Mushrooms","Parsley",0.2,"g"),
        ("Cheese & Bean JP","Large Baking Potato",1,"pcs"),("Cheese & Bean JP","Baked Beans",80,"g"),
        ("Cheese & Bean JP","Mature Grated Cheddar",40,"g"),("Cheese & Bean JP","Babyleaf Salad",15,"g"),
        ("Cheese & Bean JP","Cherry Tomatoes",1,"pcs"),
        ("Cheese JP","Large Baking Potato",1,"pcs"),("Cheese JP","Mature Grated Cheddar",40,"g"),
        ("Cheese JP","Babyleaf Salad",15,"g"),("Cheese JP","Cherry Tomatoes",1,"pcs"),
        ("Chick Flatbread","Greek Style Flatbread",1,"pcs"),("Chick Flatbread","Chicken Burger Fillet",1,"pcs"),
        ("Chick Flatbread","Unsmoked Streaky Bacon",2,"rashers"),("Chick Flatbread","Caesar Dressing",15,"g"),
        ("Chick Flatbread","Little Gem Lettuce",20,"g"),("Chick Flatbread","Parmigiano Reggiano",10,"g"),
        ("Chicken Club","White Loaf With Sourdough",3,"slices"),
        ("Chicken Club","Flamegrilled Chicken Breast",45,"g"),("Chicken Club","Light Mayonnaise",20,"g"),
        ("Chicken Club","Unsmoked Streaky Bacon",2,"rashers"),("Chicken Club","Little Gem Lettuce",2,"pcs"),
        ("Chicken Club","Salad Tomatoes",4,"slices"),("Chicken Club","Tomato Chutney",20,"g"),
        ("Chicken Club","Skin on Fries",200,"g"),("Chicken Club","Chip Seasoning",0.5,"g"),
        ("Chilli Carne JP","Large Baking Potato",1,"pcs"),("Chilli Carne JP","Chilli Con Carne",196,"g"),
        ("Chilli Carne JP","Mature Grated Cheddar",40,"g"),("Chilli Carne JP","Babyleaf Salad",15,"g"),
        ("Chilli Carne JP","Cherry Tomatoes",1,"pcs"),
        ("Egg Bacon Brioch","Brioche Bun",1,"pcs"),("Egg Bacon Brioch","Fried Egg",1,"pcs"),
        ("Egg Bacon Brioch","Unsmoked Streaky Bacon",2,"rashers"),("Egg Bacon Brioch","Wild Rocket",5,"g"),
        ("Egg Bacon Brioch","Sweet Chilli Jam",10,"g"),("Egg Bacon Brioch","Fresh Cut Coriander",2,"g"),
        ("Egg Bacon Brioch","Light Soft Cheese",35,"g"),("Egg Bacon Brioch","Crushed Chillies",0.5,"g"),
        ("Egg Bap","Large White Bap",1,"pcs"),("Egg Bap","Unsalted Butter",10,"g"),("Egg Bap","Fried Egg",2,"pcs"),
        ("F.Eggs on Toast","White Farmhouse Bread",2,"slices"),("F.Eggs on Toast","Unsalted Butter",10,"g"),
        ("F.Eggs on Toast","Fried Egg",2,"pcs"),
        ("Festive Stack","Ciabatta Roll",1,"pcs"),("Festive Stack","Chicken Burger Fillet",1,"pcs"),
        ("Festive Stack","Unsmoked Streaky Bacon",2,"rashers"),("Festive Stack","French Brie",50,"g"),
        ("Festive Stack","Cranberry Sauce",10,"g"),("Festive Stack","Little Gem Lettuce",2,"pcs"),
        ("Festive Stack","Parmigiano Reggiano",10,"g"),("Festive Stack","Skin on Fries",200,"g"),
        ("Fried Egg","Fried Egg",1,"pcs"),("Hash Brown","Hash Brown",2,"pcs"),
        ("Hash Brown Bites","Hash Brown Bites",150,"g"),
        ("Little Avo Toast","50/50 Medium Bread",1,"pcs"),("Little Avo Toast","Avocado",0.5,"pcs"),
        ("Little Avo Toast","Lime",0.25,"pcs"),
        ("Little Bean Toas","50/50 Medium Bread",1,"pcs"),("Little Bean Toas","Unsalted Butter",5,"g"),
        ("Little Bean Toas","Baked Beans",80,"g"),
        ("Ltle Bfast Bacon","Baked Beans",80,"g"),("Ltle Bfast Bacon","50/50 Medium Bread",1,"pcs"),
        ("Ltle Bfast Bacon","Scrambled Egg",1,"pcs"),("Ltle Bfast Bacon","Back Bacon",1,"pcs"),
        ("Ltle Bfast Saus","Baked Beans",80,"g"),("Ltle Bfast Saus","Scrambled Egg",1,"pcs"),
        ("Ltle Bfast Saus","50/50 Medium Bread",1,"pcs"),("Ltle Bfast Saus","Pork Sausage",1,"pcs"),
        ("Mini Hash Browns","Hash Brown Bites",150,"g"),
        ("P.Eggs on Toast","White Farmhouse Bread",2,"slices"),("P.Eggs on Toast","Unsalted Butter",10,"g"),
        ("P.Eggs on Toast","Poached Egg",2,"pcs"),("Poached Egg","Poached Egg",1,"pcs"),
        ("Posh Beans","Sourdough Loaf",1,"slices"),("Posh Beans","Five Beans in Tomato Sauce",200,"g"),
        ("Posh Beans","BBQ Sauce",15,"g"),("Posh Beans","Poached Egg",1,"pcs"),
        ("Posh Beans","Fresh Cut Coriander",1,"g"),("Roll & Butter","Large White Bap",1,"pcs"),
        ("S.Eggs on Toast","White Farmhouse Bread",2,"pcs"),("S.Eggs on Toast","Scrambled Egg",2,"pcs"),
        ("S.Eggs on Toast","Unsalted Butter",10,"g"),
        ("Sausage","Pork Sausage",1,"pcs"),("Sausage Bap","Pork Sausage",2,"pcs"),
        ("Sausage Bap","Large White Bap",1,"pcs"),("Sausage Bap","Unsalted Butter",10,"g"),
        ("Scrambled Egg","Scrambled Egg",1,"pcs"),("Scrambled Egg","Semi Skimmed Milk",20,"g"),
        ("Streaky Bacon","Unsmoked Streaky Bacon",1,"rashers"),("Tattie Scone","Tattie Scone",1,"pcs"),
        ("The Breakfast","Baked Beans",80,"g"),("The Breakfast","Pork Sausage",1,"pcs"),
        ("The Breakfast","Back Bacon",1,"rashers"),("The Breakfast","Fried Egg",1,"pcs"),
        ("The Breakfast","Hash Brown",2,"pcs"),("The Breakfast","Tattie Scone",1,"pcs"),
        ("Toasted Teacake","Toasted Teacake",1,"pcs"),
        ("Tuna JP","Large Baking Potato",1,"pcs"),("Tuna JP","Tuna Mayo Mix",80,"g"),
        ("Tuna JP","Babyleaf Salad",15,"g"),("Tuna JP","Cherry Tomatoes",1,"pcs"),
        ("Tuna Mayo Mix","Tuna",80,"g"),("Tuna Mayo Mix","Spring Onions",1,"pcs"),
        ("Tuna Mayo Mix","Light Mayonnaise",40,"g"),
        ("Tuna Melt Panini","Panini",2,"slices"),("Tuna Melt Panini","Mature Grated Cheddar",32.5,"g"),
        ("Tuna Melt Panini","Tuna Mayo Mix",80,"g"),
        ("Tuna Panini","Panini",2,"slices"),("Tuna Panini","Mature Grated Cheddar",32.5,"g"),
        ("Tuna Panini","Tuna Mayo Mix",80,"g"),
        ("Tuna Toastie","Sourdough Loaf",2,"slices"),("Tuna Toastie","Mature Grated Cheddar",32.5,"g"),
        ("Tuna Toastie","Grated Mozzarella",12.5,"g"),("Tuna Toastie","Bechamel",25,"g"),
        ("Tuna Toastie","Tuna Mayo Mix",25,"g"),
        ("Veg Sausage Bap","Large White Bap",1,"pcs"),("Veg Sausage Bap","Unsalted Butter",10,"g"),
        ("Veg Sausage Bap","Vegan Sausages",2,"pcs"),
        ("Vegan Breakfast","Baked Beans",80,"g"),("Vegan Breakfast","Avocado",0.5,"pcs"),
        ("Vegan Breakfast","Salad Tomato",0.5,"pcs"),("Vegan Breakfast","Large Flat Mushrooms",1,"pcs"),
        ("Vegan Breakfast","Multiseed Bread",1,"slices"),("Vegan Breakfast","Hash Brown",1,"pcs"),
        ("Vegan Breakfast","Vegan Sausages",2,"pcs"),("Vegan Breakfast","Lime",0.25,"pcs"),
        ("Vegan Sausage","Vegan Sausage",1,"pcs"),
        ("Veggie Bap","Large White Bap",1,"pcs"),("Veggie Bap","Unsalted Butter",10,"g"),
        ("Veggie Bap","Vegan Sausages",2,"pcs"),
        ("Veggie Breakfast","Baked Beans",80,"g"),("Veggie Breakfast","Vegan Sausages",2,"pcs"),
        ("Veggie Breakfast","Fried Egg",1,"pcs"),("Veggie Breakfast","Hash Brown",2,"pcs"),
        ("Veggie Breakfast","Tattie Scone",1,"pcs"),
        ("Bakery","Bakery",1,"pcs"),("White Toast Bread","White Farmhouse Bread",2,"pcs"),
        ("Brown Toast Bread","Brown Farmhouse Bread",2,"pcs"),("Porridge","Porridge Pot",1,"pcs"),
        ("Sourdough Toast Bread","Sourdough Loaf",2,"pcs"),
        ("Multiseed Toast Bread","Multiseed Bread",2,"pcs"),
        ("Little Egg Toast","50/50 Medium Bread",1,"pcs"),("Little Egg Toast","Fried Egg",1,"pcs"),
        ("Extra Beans","Baked Beans",80,"g"),
    ]
    return pd.DataFrame(data, columns=['product','ingredient','quantity','unit'])


# ── pick the best model from the DB ──

def select_best_models(db_path):
    # Check model_tracking.db for the winning model at each horizon.
    if not os.path.exists(db_path):
        print(f"  WARNING: {db_path} not found, will default to xgb_improved_daily")
        return None

    conn = sqlite3.connect(db_path)
    best = {}

    print("\n  Model selection (from model_tracking.db):")
    for hkey, db_label in [('day','1-Day'), ('week','1-Week'), ('month','1-Month')]:
        rows = conn.execute("""
            SELECT model_type, WAPE, run_id, MASE, MAE, Bias
            FROM metrics_summary
            WHERE product_name='ALL_PRODUCTS' AND evaluation_horizon=?
              AND WAPE IS NOT NULL
            ORDER BY WAPE ASC LIMIT 5
        """, (db_label,)).fetchall()

        if rows:
            mt, wape, rid, mase, mae, bias = rows[0]
            best[hkey] = dict(model_type=mt, wape=wape, run_id=rid, mase=mase, mae=mae, bias=bias)
            runner_up = f" (runner-up: {rows[1][0]} @ {rows[1][1]:.4f})" if len(rows) > 1 else ""
            print(f"    {db_label:>7s} → {mt}  WAPE={wape:.4f}{runner_up}")
        else:
            print(f"    {db_label:>7s} → no data yet")
            best[hkey] = None

    conn.close()
    return best


def resolve_runner(model_type):
    # Figure out which runner function to call for a given DB model_type string.
    runner = MODEL_TYPE_TO_RUNNER.get(model_type)
    if runner:
        return runner
    # fuzzy match for model types I haven't explicitly mapped
    # we do it like this because the models are saved in db with date appended in name
    model_type_lower = model_type.lower()
    if 'xgboost' in model_type_lower and 'simple' in model_type_lower and 'hourly' in model_type_lower:
        return 'xgb_simple_hourly'
    if 'xgboost' in model_type_lower and 'hourly' in model_type_lower:
        return 'xgb_improved_hourly'
    if 'xgboost' in model_type_lower and 'simple' in model_type_lower:
        return 'xgb_simple_daily'
    if 'xgboost' in model_type_lower:
        return 'xgb_improved_daily'
    if 'lstm_forcast' in model_type_lower and 'hourly' in model_type_lower:
        return 'lstm_hourly'
    if 'lstm_forcast' in model_type_lower:
        return 'lstm_daily'
    if 'prophet' in model_type_lower and 'hourly' in model_type_lower:
        return 'prophet_hourly'
    if 'prophet' in model_type_lower:
        return 'prophet_daily'
    if 'arima_forecast' in model_type_lower:
        return 'arima_forecast'
    print(f"  WARNING: don't recognise model_type '{model_type}', falling back to xgb_improved_daily")
    return 'xgb_improved_daily'


# ── unit conversion ──
# anything over 500g gets shown as kg in the reports for a more easy visualization, rather than having 7850g you would get 7.85kg

def convert_grams_to_kg(df):
    out = df.copy()
    big = (out['unit'] == 'g') & (out['total_qty'] >= 500)
    out.loc[big, 'total_qty'] = out.loc[big, 'total_qty'] / 1000.0
    out.loc[big, 'unit'] = 'kg'
    return out

def fmt_qty(q, u):
    if u == 'kg':
        return f"{q:.2f} kg"
    if q == int(q):
        return f"{int(q)} {u}"
    return f"{q:.1f} {u}"


# ── data loading (shared by all models) ──

def load_sales_long():
    sales_data = pd.read_csv(os.path.join(DATA_DIR, 'sales_data_preprocessed.csv'))
    sales_data['Date'] = pd.to_datetime(sales_data['Date']).dt.normalize()
    if 'Time' in sales_data.columns:
        sales_data = sales_data.drop(columns=['Time'])
    daily_sales = sales_data.groupby('Date').sum(numeric_only=True).reset_index()
    product_cols = [col for col in daily_sales.columns if col != 'Date']

    df_long = pd.melt(daily_sales, id_vars=['Date'], value_vars=product_cols,
                      var_name='Product_Name', value_name='Sales')
    df_long['Sales'] = df_long['Sales'].clip(lower=0)
    df_long = df_long[df_long['Product_Name'].isin(PRODUCTS_TO_FORECAST)].reset_index(drop=True)
    df_long = df_long.sort_values(['Product_Name', 'Date']).reset_index(drop=True)
    return df_long, daily_sales

def load_sales_hourly():
    # Load hourly sales filtered to business hours only (8-16).
    sales_data = pd.read_csv(os.path.join(DATA_DIR, 'sales_data_preprocessed.csv'))
    sales_data['Date'] = pd.to_datetime(sales_data['Date'])
    sales_data = sales_data[sales_data['Date'].dt.hour.isin(BUSINESS_HOURS)]
    if 'Time' in sales_data.columns:
        sales_data = sales_data.drop(columns=['Time'])
    product_cols = [col for col in sales_data.columns if col not in ['Date', 'Time', 'date'] and sales_data[col].dtype.kind in 'iufc']

    df_long = pd.melt(sales_data, id_vars=['Date'], value_vars=product_cols,
                      var_name='Product_Name', value_name='Sales')
    df_long['Sales'] = df_long['Sales'].clip(lower=0)
    df_long = df_long[df_long['Product_Name'].isin(PRODUCTS_TO_FORECAST)].reset_index(drop=True)
    return df_long.sort_values(['Product_Name', 'Date']).reset_index(drop=True)

def load_exogenous():
    # weather
    weather_data = pd.read_csv(os.path.join(DATA_DIR, 'weather_data_hourly.csv'))
    weather_data['Date'] = pd.to_datetime(weather_data['Date']).dt.normalize()
    aggregation_map = {'apparent_temperature':'mean', 'precipitation':'sum', 'snowfall':'sum',
                       'snow_depth':'max', 'relative_humidity_2m':'mean', 'cloud_cover':'mean',
                       'visibility':'mean', 'wind_speed_10m':'mean', 'wind_gusts_10m':'max'}
    if 'weather_code' in weather_data.columns:
        weather_data['is_clear'] = (weather_data['weather_code'] == 0).astype(int)
        weather_data['is_cloudy'] = weather_data['weather_code'].isin([1,2,3,45,48]).astype(int)
        weather_data['is_rain'] = weather_data['weather_code'].isin([51,53,55,56,57,61,63,65,66,67,80,81,82,95,96,99]).astype(int)
        weather_data['is_snow'] = weather_data['weather_code'].isin([71,73,75,77,85,86]).astype(int)
        aggregation_map.update({'is_clear':'max', 'is_cloudy':'max', 'is_rain':'max', 'is_snow':'max'})
    daily_weather = weather_data.groupby('Date').agg(aggregation_map).reset_index()

    # holidays
    holiday_data = pd.read_csv(os.path.join(DATA_DIR, 'holidays_data_preprocessed.csv'))
    holiday_data['Date'] = pd.to_datetime(holiday_data['Date']).dt.normalize()
    daily_holidays = holiday_data.groupby('Date').max().reset_index()
    daily_holidays['is_holiday_lag_1'] = daily_holidays['is_holiday'].shift(1).fillna(0)
    daily_holidays['is_holiday_lead_1'] = daily_holidays['is_holiday'].shift(-1).fillna(0)

    # events
    event_data = pd.read_csv(os.path.join(DATA_DIR, 'aberdeen_events_master_timeline.csv'))
    event_data['Date'] = pd.to_datetime(event_data['Date']).dt.normalize()
    daily_events = event_data.groupby('Date').max(numeric_only=True).reset_index()

    return daily_weather, daily_holidays, daily_events


# ══════════════════════════════════════════
# MODEL RUNNERS
# each returns a DataFrame: Date, Product_Name, Forecast
# all of these are a more compact version found in the notebooks
# ══════════════════════════════════════════

# ── 1. ARIMA ──
def run_arima(forecast_days, eval_mode=None):
    from statsforecast import StatsForecast
    from statsforecast.models import ARIMA as AM
    print("    Loading data...")
    df_long, _ = load_sales_long()
    
    if eval_mode == 'november':
        val_end = '2025-11-01'
        time_series_data = df_long[df_long['Date'] <= val_end].rename(columns={'Product_Name':'unique_id','Date':'ds','Sales':'y'})
        print(f"    Evaluating November 2025 (Training up to {val_end})...")
    else:
        time_series_data = df_long.rename(columns={'Product_Name':'unique_id','Date':'ds','Sales':'y'})
        print(f"    Training ARIMA(1,1,1) → {forecast_days} days...")
    
    stats_forecast = StatsForecast(models=[AM(order=(1,1,1),seasonal_order=(0,0,0),season_length=1,alias='ARIMA')],freq='D',n_jobs=-1)
    forecast_results = stats_forecast.forecast(df=time_series_data[['unique_id','ds','y']], h=forecast_days)
    forecast_results['ARIMA'] = forecast_results['ARIMA'].clip(lower=0).round().astype(int)
    return forecast_results.rename(columns={'unique_id':'Product_Name','ds':'Date','ARIMA':'Forecast'})[['Date','Product_Name','Forecast']].reset_index(drop=True)

# ── 2. PROPHET DAILY ──
def run_prophet_daily(forecast_days, eval_mode=None):
    from prophet import Prophet
    import logging as lg
    lg.getLogger('prophet').setLevel(lg.WARNING)
    lg.getLogger('cmdstanpy').setLevel(lg.WARNING)
    print("    Loading data...")
    df_long, _ = load_sales_long()
    daily_weather, daily_holidays, daily_events = load_exogenous()
    
    if eval_mode == 'november':
        val_end = '2025-11-01'
        df_long = df_long[df_long['Date'] <= val_end]
        print(f"    Evaluating November 2025 (Prophet Daily, Train <= {val_end})...")

    products = sorted(df_long['Product_Name'].unique())
    results = []
    print(f"    Training Prophet for {len(products)} products → {forecast_days} days...")
    for product in products:
        product_df = df_long[df_long['Product_Name']==product][['Date','Sales']].rename(columns={'Date':'ds','Sales':'y'})
        product_df = product_df.merge(daily_holidays[['Date','is_holiday']].rename(columns={'Date':'ds'}), on='ds', how='left').fillna(0)
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                        seasonality_mode='additive', changepoint_prior_scale=0.05)
        model.add_country_holidays(country_name='GB')
        model.fit(product_df[['ds','y']])
        future_df = model.make_future_dataframe(periods=forecast_days)
        forecast_df = model.predict(future_df)
        forecast_df = forecast_df.tail(forecast_days)[['ds','yhat']].copy()
        forecast_df['yhat'] = forecast_df['yhat'].clip(lower=0).round().astype(int)
        forecast_df['Product_Name'] = product
        results.append(forecast_df)
    output_df = pd.concat(results, ignore_index=True)
    return output_df.rename(columns={'ds':'Date','yhat':'Forecast'})[['Date','Product_Name','Forecast']]

# ── 3. PROPHET HOURLY ──
def run_prophet_hourly(forecast_days, eval_mode=None):
    from prophet import Prophet
    import logging as lg
    lg.getLogger('prophet').setLevel(lg.WARNING)
    lg.getLogger('cmdstanpy').setLevel(lg.WARNING)
    print("    Loading hourly data...")
    df_long = load_sales_hourly()
    
    if eval_mode == 'november':
        val_end = '2025-11-01 23:59:59'
        df_long = df_long[df_long['Date'] <= val_end]
        print(f"    Evaluating November 2025 (Hourly Prophet, Train <= {val_end})...")

    products = sorted(df_long['Product_Name'].unique())
    results = []
    forecast_hours = forecast_days * HOURS_PER_DAY
    print(f"    Training Hourly Prophet for {len(products)} products → {forecast_days} days ({forecast_hours} hours)...")
    for product in products:
        product_df = df_long[df_long['Product_Name']==product][['Date','Sales']].rename(columns={'Date':'ds','Sales':'y'})
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                        seasonality_mode='additive', changepoint_prior_scale=0.05)
        model.add_seasonality(name='daily_business', period=1, fourier_order=5)
        model.add_country_holidays(country_name='GB')
        model.fit(product_df[['ds','y']])
        future_df = model.make_future_dataframe(periods=forecast_hours, freq='h')
        future_df = future_df[future_df['ds'].dt.hour.isin(BUSINESS_HOURS)]
        forecast_df = model.predict(future_df)
        forecast_df = forecast_df.tail(forecast_hours)[['ds','yhat']].copy()
        forecast_df['yhat'] = forecast_df['yhat'].clip(lower=0)
        forecast_df['Product_Name'] = product
        results.append(forecast_df)
    # Rollup hourly → daily
    combined_results = pd.concat(results, ignore_index=True)
    combined_results['Date'] = combined_results['ds'].dt.normalize()
    daily_results = combined_results.groupby(['Date','Product_Name'])['yhat'].sum().reset_index()
    daily_results['Forecast'] = daily_results['yhat'].round().astype(int)
    return daily_results[['Date','Product_Name','Forecast']]

# ── 4. XGB IMPROVED DAILY (same as v2) ──
def run_xgb_improved_daily(forecast_days, eval_mode=None):
    import xgboost as xgb
    import optuna; optuna.logging.set_verbosity(logging.WARNING)
    print("    Loading data...")
    df_long, _ = load_sales_long(); daily_weather, daily_holidays, daily_events = load_exogenous()
    df_long['Product_Name'] = df_long['Product_Name'].astype('category')
    # Time features
    df_long['day_of_week'] = df_long['Date'].dt.dayofweek
    df_long['day_sin'] = np.sin(2 * np.pi * df_long['day_of_week'] / 7)
    df_long['day_cos'] = np.cos(2 * np.pi * df_long['day_of_week'] / 7)
    df_long['month'] = df_long['Date'].dt.month
    df_long['month_sin'] = np.sin(2 * np.pi * (df_long['month'] - 1) / 12)
    df_long['month_cos'] = np.cos(2 * np.pi * (df_long['month'] - 1) / 12)
    df_long['day_of_month'] = df_long['Date'].dt.day
    df_long['Is_Weekend'] = df_long['Date'].dt.dayofweek.isin([5, 6]).astype(int)
    df_long['Year'] = df_long['Date'].dt.year
    df_long['week_of_year'] = df_long['Date'].dt.isocalendar().week.astype(int)
    df_merged = df_long.merge(daily_weather, on='Date', how='left').merge(daily_holidays, on='Date', how='left').merge(daily_events, on='Date', how='left')
    for col in ['date', 'Time']:
        if col in df_merged.columns:
            df_merged = df_merged.drop(columns=[col])
    categorical_cols = df_merged.select_dtypes(include=['category']).columns
    df_merged[df_merged.columns.difference(categorical_cols)] = df_merged[df_merged.columns.difference(categorical_cols)].fillna(0)
    df_merged = df_merged.sort_values(['Product_Name', 'Date']).reset_index(drop=True)
    for lag in [1, 2, 7, 14, 30]:
        df_merged[f'sales_{lag}_step_ago'] = df_merged.groupby('Product_Name', observed=False)['Sales'].shift(lag)
    for window in [3, 7, 14]:
        df_merged[f'rolling_{window}d_avg'] = df_merged.groupby('Product_Name', observed=False)['sales_1_step_ago'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df_merged[f'rolling_{window}d_std'] = df_merged.groupby('Product_Name', observed=False)['sales_1_step_ago'].transform(lambda x: x.rolling(window, min_periods=1).std()).fillna(0)
    df_merged['sales_momentum'] = df_merged['sales_1_step_ago'] - df_merged['sales_7_step_ago']
    df_merged['expanding_mean'] = df_merged.groupby('Product_Name', observed=False)['sales_1_step_ago'].transform(lambda x: x.expanding(min_periods=1).mean())
    df_merged['ratio_1d_vs_7d'] = df_merged['sales_1_step_ago'] / (df_merged['rolling_7d_avg'] + 1e-8)
    df_merged = df_merged.dropna().reset_index(drop=True)
    df_merged['Sales'] = df_merged['Sales'].clip(lower=0)
    feature_cols = [c for c in df_merged.columns if c not in ['Date', 'Sales']]
    
    if eval_mode == 'november':
        train_end = pd.to_datetime('2025-10-01')
        val_end = pd.to_datetime('2025-11-01')
        X_train = df_merged[df_merged['Date'] <= train_end][feature_cols]
        y_train = df_merged[df_merged['Date'] <= train_end]['Sales']
        X_val = df_merged[(df_merged['Date'] > train_end) & (df_merged['Date'] <= val_end)][feature_cols]
        y_val = df_merged[(df_merged['Date'] > train_end) & (df_merged['Date'] <= val_end)]['Sales']
        print(f"    Evaluating November 2025 (Train <= {train_end}, Val <= {val_end})...")
    else:
        max_date = df_merged['Date'].max()
        val_start = max_date - pd.Timedelta(days=30)
        X_train = df_merged[df_merged['Date'] <= val_start][feature_cols]
        y_train = df_merged[df_merged['Date'] <= val_start]['Sales']
        X_val = df_merged[df_merged['Date'] > val_start][feature_cols]
        y_val = df_merged[df_merged['Date'] > val_start]['Sales']

    print(f"    Tuning XGBoost Improved ({OPTUNA_TRIALS} trials)...")
    def objective(trial):
        params = {"n_estimators": 1500, "early_stopping_rounds": 50, "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.1, log=True),
                  "max_depth": trial.suggest_int("max_depth", 4, 8), "min_child_weight": trial.suggest_int("min_child_weight", 2, 8),
                  "subsample": trial.suggest_float("subsample", 0.6, 0.95), "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                  "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True), "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 5.0, log=True),
                  "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 5.0, log=True), "enable_categorical": True, "tree_method": "hist"}
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return np.sqrt(np.mean((y_val - model.predict(X_val)) ** 2))
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    best_params = study.best_params
    best_params.update({"n_estimators": 1500, "early_stopping_rounds": 30, "enable_categorical": True, "tree_method": "hist"})
    best_params.pop('early_stopping_rounds', None)
    print(f"    Best RMSE: {study.best_value:.4f}. Training final model...")
    
    if eval_mode == 'november':
        val_end = pd.to_datetime('2025-11-01')
        full_df = df_merged[df_merged['Date'] <= val_end]
        model = xgb.XGBRegressor(**best_params)
        model.fit(full_df[feature_cols], full_df['Sales'], verbose=False)
        forecast_start_df = full_df
    else:
        model = xgb.XGBRegressor(**best_params)
        model.fit(df_merged[feature_cols], df_merged['Sales'], verbose=False)
        forecast_start_df = df_merged

    # Recursive forecast
    return _xgb_recursive_forecast(forecast_start_df, model, feature_cols, forecast_days,
        lag_map={1:'sales_1_step_ago',2:'sales_2_step_ago',7:'sales_7_step_ago',14:'sales_14_step_ago',30:'sales_30_step_ago'},
        rolling_windows=[3,7,14], use_momentum=True)

# ── 5. XGB SIMPLE DAILY ──
def run_xgb_simple_daily(forecast_days, eval_mode=None):
    import xgboost as xgb
    import optuna; optuna.logging.set_verbosity(logging.WARNING)
    print("    Loading data...")
    df_long, _ = load_sales_long(); daily_weather, daily_holidays, daily_events = load_exogenous()
    df_long['Product_Name'] = df_long['Product_Name'].astype('category')
    df_merged = df_long.merge(daily_weather, on='Date', how='left').merge(daily_holidays, on='Date', how='left').merge(daily_events, on='Date', how='left')
    for col in ['date', 'Time']:
        if col in df_merged.columns:
            df_merged = df_merged.drop(columns=[col])
    categorical_cols = df_merged.select_dtypes(include=['category']).columns
    df_merged[df_merged.columns.difference(categorical_cols)] = df_merged[df_merged.columns.difference(categorical_cols)].fillna(0)
    df_merged = df_merged.sort_values(['Product_Name', 'Date']).reset_index(drop=True)
    # Simple lags: 1, 7, 30 only
    for lag, name in [(1, 'sales_1_step_ago'), (7, 'sales_7_steps_ago'), (30, 'sales_30_steps_ago')]:
        df_merged[name] = df_merged.groupby('Product_Name', observed=False)['Sales'].shift(lag)
    df_merged = df_merged.dropna().reset_index(drop=True)
    df_merged['Sales'] = df_merged['Sales'].clip(lower=0)
    feature_cols = [c for c in df_merged.columns if c not in ['Date', 'Sales']]
    
    if eval_mode == 'november':
        train_end = pd.to_datetime('2025-10-01')
        val_end = pd.to_datetime('2025-11-01')
        X_train = df_merged[df_merged['Date'] <= train_end][feature_cols]
        y_train = df_merged[df_merged['Date'] <= train_end]['Sales']
        X_val = df_merged[(df_merged['Date'] > train_end) & (df_merged['Date'] <= val_end)][feature_cols]
        y_val = df_merged[(df_merged['Date'] > train_end) & (df_merged['Date'] <= val_end)]['Sales']
        print(f"    Evaluating November 2025 (Train <= {train_end}, Val <= {val_end})...")
    else:
        max_date = df_merged['Date'].max()
        val_start = max_date - pd.Timedelta(days=30)
        X_train = df_merged[df_merged['Date'] <= val_start][feature_cols]
        y_train = df_merged[df_merged['Date'] <= val_start]['Sales']
        X_val = df_merged[df_merged['Date'] > val_start][feature_cols]
        y_val = df_merged[df_merged['Date'] > val_start]['Sales']

    print(f"    Tuning XGBoost Simple ({OPTUNA_TRIALS} trials)...")
    def objective(trial):
        params = {"n_estimators": 1000, "early_stopping_rounds": 50, "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.1, log=True),
                  "max_depth": trial.suggest_int("max_depth", 3, 7), "min_child_weight": trial.suggest_int("min_child_weight", 2, 8),
                  "subsample": trial.suggest_float("subsample", 0.6, 0.95), "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                  "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True), "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 5.0, log=True),
                  "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 5.0, log=True), "enable_categorical": True, "tree_method": "hist"}
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return np.sqrt(np.mean((y_val - model.predict(X_val)) ** 2))
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    best_params = study.best_params
    best_params.update({"n_estimators": 1000, "early_stopping_rounds": 30, "enable_categorical": True, "tree_method": "hist"})
    best_params.pop('early_stopping_rounds', None)
    print(f"    Best RMSE: {study.best_value:.4f}. Training final model...")
    
    if eval_mode == 'november':
        val_end = pd.to_datetime('2025-11-01')
        full_df = df_merged[df_merged['Date'] <= val_end]
        model = xgb.XGBRegressor(**best_params)
        model.fit(full_df[feature_cols], full_df['Sales'], verbose=False)
        forecast_start_df = full_df
    else:
        model = xgb.XGBRegressor(**best_params)
        model.fit(df_merged[feature_cols], df_merged['Sales'], verbose=False)
        forecast_start_df = df_merged

    return _xgb_recursive_forecast(forecast_start_df, model, feature_cols, forecast_days,
        lag_map={1:'sales_1_step_ago',7:'sales_7_steps_ago',30:'sales_30_steps_ago'},
        rolling_windows=[], use_momentum=False)

# ── SHARED XGB RECURSIVE FORECAST HELPER ──
def _xgb_recursive_forecast(df, model, feature_cols, forecast_days, lag_map, rolling_windows, use_momentum):
    # Shared recursive daily XGBoost forecast logic.
    last_date = df['Date'].max()
    forecast_dates = [last_date + pd.Timedelta(days=i+1) for i in range(forecast_days)]
    products = sorted(PRODUCTS_TO_FORECAST)
    sales_history = {(row['Product_Name'], row['Date']): row['Sales'] for _, row in df[['Date', 'Product_Name', 'Sales']].iterrows()}
    forecast_rows = [{'Date': d, 'Product_Name': p} for d in forecast_dates for p in products]
    forecast_df = pd.DataFrame(forecast_rows)
    forecast_df['Product_Name'] = forecast_df['Product_Name'].astype(pd.CategoricalDtype(categories=df['Product_Name'].cat.categories))
    # Copy time features
    for time_col in ['day_of_week', 'day_sin', 'day_cos', 'month', 'month_sin', 'month_cos', 'day_of_month', 'Is_Weekend', 'Year', 'week_of_year']:
        if time_col in feature_cols:
            if time_col == 'day_of_week':
                forecast_df[time_col] = forecast_df['Date'].dt.dayofweek
            elif time_col == 'day_sin':
                forecast_df[time_col] = np.sin(2 * np.pi * forecast_df['Date'].dt.dayofweek / 7)
            elif time_col == 'day_cos':
                forecast_df[time_col] = np.cos(2 * np.pi * forecast_df['Date'].dt.dayofweek / 7)
            elif time_col == 'month':
                forecast_df[time_col] = forecast_df['Date'].dt.month
            elif time_col == 'month_sin':
                forecast_df[time_col] = np.sin(2 * np.pi * (forecast_df['Date'].dt.month - 1) / 12)
            elif time_col == 'month_cos':
                forecast_df[time_col] = np.cos(2 * np.pi * (forecast_df['Date'].dt.month - 1) / 12)
            elif time_col == 'day_of_month':
                forecast_df[time_col] = forecast_df['Date'].dt.day
            elif time_col == 'Is_Weekend':
                forecast_df[time_col] = forecast_df['Date'].dt.dayofweek.isin([5, 6]).astype(int)
            elif time_col == 'Year':
                forecast_df[time_col] = forecast_df['Date'].dt.year
            elif time_col == 'week_of_year':
                forecast_df[time_col] = forecast_df['Date'].dt.isocalendar().week.astype(int)
    # Exogenous: last known
    exog_exclude = set(['Date', 'Sales', 'Product_Name', 'day_of_week', 'day_sin', 'day_cos', 'month',
                        'month_sin', 'month_cos', 'day_of_month', 'Is_Weekend', 'Year', 'week_of_year'])
    exog_exclude.update(lag_map.values())
    for window_size in rolling_windows:
        exog_exclude.update([f'rolling_{window_size}d_avg', f'rolling_{window_size}d_std'])
    if use_momentum:
        exog_exclude.update(['sales_momentum', 'expanding_mean', 'ratio_1d_vs_7d'])
    exog_cols = [c for c in df.columns if c not in exog_exclude and c in feature_cols]
    if exog_cols:
        last_exog = df[df['Date'] == last_date][exog_cols].iloc[0]
        for c in exog_cols:
            forecast_df[c] = last_exog[c]
    # Init lags
    for row_index, row in forecast_df.iterrows():
        product_name, current_date = row['Product_Name'], row['Date']
        for lag_days, lag_col in lag_map.items():
            forecast_df.at[row_index, lag_col] = sales_history.get((product_name, current_date - pd.Timedelta(days=lag_days)), 0)
        if rolling_windows or use_momentum:
            recent = [sales_history.get((product_name, current_date - pd.Timedelta(days=i)), 0) for i in range(1, 15)]
            for window_size in rolling_windows:
                window = recent[:window_size]
                forecast_df.at[row_index, f'rolling_{window_size}d_avg'] = np.mean(window)
                forecast_df.at[row_index, f'rolling_{window_size}d_std'] = np.std(window, ddof=1) if len(window) > 1 else 0
            if use_momentum:
                forecast_df.at[row_index, 'sales_momentum'] = forecast_df.at[row_index, list(lag_map.values())[0]] - forecast_df.at[row_index, lag_map.get(7, list(lag_map.values())[-1])]
                forecast_df.at[row_index, 'expanding_mean'] = np.mean(recent)
                r7 = forecast_df.at[row_index, 'rolling_7d_avg'] if 'rolling_7d_avg' in forecast_df.columns else 0
                forecast_df.at[row_index, 'ratio_1d_vs_7d'] = forecast_df.at[row_index, list(lag_map.values())[0]] / (r7 + 1e-8)
    for c in feature_cols:
        if c not in forecast_df.columns:
            forecast_df[c] = 0
    for c in forecast_df.select_dtypes(include=[np.number]).columns:
        forecast_df[c] = forecast_df[c].fillna(0)
    idx_lookup = {(r['Product_Name'], r['Date']): i for i, r in forecast_df.iterrows()}
    for day_index, current_day in enumerate(forecast_dates):
        day_indices = forecast_df.index[forecast_df['Date'] == current_day].tolist()
        preds = np.clip(model.predict(forecast_df.loc[day_indices, feature_cols]), 0, None).round().astype(int)
        forecast_df.loc[day_indices, 'Forecast'] = preds
        for row_idx, pred_value in zip(day_indices, preds):
            product = forecast_df.at[row_idx, 'Product_Name']
            sales_history[(product, current_day)] = pred_value
            for lag_days, lag_col in lag_map.items():
                future_key = (product, current_day + pd.Timedelta(days=lag_days))
                if future_key in idx_lookup:
                    forecast_df.at[idx_lookup[future_key], lag_col] = pred_value
        if day_index + 1 < len(forecast_dates):
            next_date = forecast_dates[day_index + 1]
            for next_idx in forecast_df.index[forecast_df['Date'] == next_date]:
                p = forecast_df.at[next_idx, 'Product_Name']
                recent = [sales_history.get((p, next_date - pd.Timedelta(days=i)), 0) for i in range(1, 15)]
                for window_size in rolling_windows:
                    window = recent[:window_size]
                    forecast_df.at[next_idx, f'rolling_{window_size}d_avg'] = np.mean(window)
                    forecast_df.at[next_idx, f'rolling_{window_size}d_std'] = np.std(window, ddof=1) if len(window) > 1 else 0
                if use_momentum:
                    forecast_df.at[next_idx, 'sales_momentum'] = forecast_df.at[next_idx, list(lag_map.values())[0]] - forecast_df.at[next_idx, lag_map.get(7, list(lag_map.values())[-1])]
                    forecast_df.at[next_idx, 'expanding_mean'] = np.mean(recent)
                    forecast_df.at[next_idx, 'ratio_1d_vs_7d'] = forecast_df.at[next_idx, list(lag_map.values())[0]] / (forecast_df.at[next_idx, 'rolling_7d_avg'] + 1e-8)
    result = forecast_df[['Date', 'Product_Name', 'Forecast']].copy()
    result['Forecast'] = result['Forecast'].astype(int)
    return result

# ── 6 & 7. XGB HOURLY (simple + improved) ──
def run_xgb_simple_hourly(forecast_days, eval_mode=None):
    # Simple XGBoost hourly: 3 hourly lags, predict per-hour then aggregate to daily.
    import xgboost as xgb
    import optuna; optuna.logging.set_verbosity(logging.WARNING)
    print("    Loading hourly data...")
    df_long = load_sales_hourly()
    df_long['Product_Name'] = df_long['Product_Name'].astype('category')
    df_long = df_long.sort_values(['Product_Name', 'Date']).reset_index(drop=True)
    # Simple hourly lags
    for lag, name in [(1, 'sales_1h_ago'), (HOURS_PER_DAY, 'sales_same_hour_yesterday'), (HOURS_PER_DAY * 7, 'sales_same_hour_last_week')]:
        df_long[name] = df_long.groupby('Product_Name', observed=False)['Sales'].shift(lag)
    # Time features
    df_long['hour_of_day'] = df_long['Date'].dt.hour
    df_long['day_of_week'] = df_long['Date'].dt.dayofweek
    df_long['Is_Weekend'] = df_long['day_of_week'].isin([5, 6]).astype(int)
    df_long = df_long.dropna().reset_index(drop=True)
    df_long['Sales'] = df_long['Sales'].clip(lower=0)
    feature_cols = [c for c in df_long.columns if c not in ['Date', 'Sales']]
    
    if eval_mode == 'november':
        train_end = pd.to_datetime('2025-10-01')
        val_end = pd.to_datetime('2025-11-01')
        X_train = df_long[df_long['Date'] <= train_end][feature_cols]
        y_train = df_long[df_long['Date'] <= train_end]['Sales']
        X_val = df_long[(df_long['Date'] > train_end) & (df_long['Date'] <= val_end)][feature_cols]
        y_val = df_long[(df_long['Date'] > train_end) & (df_long['Date'] <= val_end)]['Sales']
        print(f"    Evaluating November 2025 (Simple Hourly, Train <= {train_end}, Val <= {val_end})...")
    else:
        max_date = df_long['Date'].max()
        val_start = max_date - pd.Timedelta(days=30)
        X_train = df_long[df_long['Date'] <= val_start][feature_cols]
        y_train = df_long[df_long['Date'] <= val_start]['Sales']
        X_val = df_long[df_long['Date'] > val_start][feature_cols]
        y_val = df_long[df_long['Date'] > val_start]['Sales']

    print(f"    Tuning Simple Hourly XGBoost ({OPTUNA_TRIALS} trials)...")
    def objective(trial):
        params = {"n_estimators": 1000, "early_stopping_rounds": 50, "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.1, log=True),
                  "max_depth": trial.suggest_int("max_depth", 3, 7), "min_child_weight": trial.suggest_int("min_child_weight", 2, 8),
                  "subsample": trial.suggest_float("subsample", 0.6, 0.95), "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                  "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True), "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 5.0, log=True),
                  "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 5.0, log=True), "enable_categorical": True, "tree_method": "hist"}
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return np.sqrt(np.mean((y_val - model.predict(X_val)) ** 2))
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    best_params = study.best_params
    best_params.update({"n_estimators": 1000, "early_stopping_rounds": 30, "enable_categorical": True, "tree_method": "hist"})
    best_params.pop('early_stopping_rounds', None)
    print(f"    Best RMSE: {study.best_value:.4f}. Training + forecasting...")
    
    if eval_mode == 'november':
        val_end = pd.to_datetime('2025-11-01')
        full_df = df_long[df_long['Date'] <= val_end]
        model = xgb.XGBRegressor(**best_params)
        model.fit(full_df[feature_cols], full_df['Sales'], verbose=False)
        forecast_start_df = full_df
    else:
        model = xgb.XGBRegressor(**best_params)
        model.fit(df_long[feature_cols], df_long['Sales'], verbose=False)
        forecast_start_df = df_long

    return _xgb_hourly_recursive(forecast_start_df, model, feature_cols, forecast_days,
        lag_map={1: 'sales_1h_ago', HOURS_PER_DAY: 'sales_same_hour_yesterday', HOURS_PER_DAY * 7: 'sales_same_hour_last_week'},
        rolling_windows=[])

def run_xgb_improved_hourly(forecast_days, eval_mode=None):
    # Improved XGBoost hourly: 5 hourly lags, rolling, rush flags."""
    import xgboost as xgb
    import optuna; optuna.logging.set_verbosity(logging.WARNING)
    print("    Loading hourly data...")
    df_long = load_sales_hourly()
    df_long['Product_Name'] = df_long['Product_Name'].astype('category')
    df_long = df_long.sort_values(['Product_Name', 'Date']).reset_index(drop=True)
    # Improved hourly lags
    for lag, name in [(1, 'sales_1h_ago'), (2, 'sales_2h_ago'), (HOURS_PER_DAY, 'sales_same_hour_yesterday'),
                     (HOURS_PER_DAY * 7, 'sales_same_hour_last_week'), (HOURS_PER_DAY * 14, 'sales_same_hour_2weeks_ago')]:
        df_long[name] = df_long.groupby('Product_Name', observed=False)['Sales'].shift(lag)
    # Rolling
    for window, window_name in [(HOURS_PER_DAY, '1d'), (HOURS_PER_DAY * 3, '3d'), (HOURS_PER_DAY * 7, '7d')]:
        df_long[f'rolling_{window_name}_avg'] = df_long.groupby('Product_Name', observed=False)['sales_1h_ago'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df_long[f'rolling_{window_name}_std'] = df_long.groupby('Product_Name', observed=False)['sales_1h_ago'].transform(lambda x: x.rolling(window, min_periods=1).std()).fillna(0)
    # Time + rush features
    df_long['hour_of_day'] = df_long['Date'].dt.hour
    df_long['hour_sin'] = np.sin(2 * np.pi * (df_long['hour_of_day'] - 8) / 9)
    df_long['hour_cos'] = np.cos(2 * np.pi * (df_long['hour_of_day'] - 8) / 9)
    df_long['day_of_week'] = df_long['Date'].dt.dayofweek
    df_long['day_sin'] = np.sin(2 * np.pi * df_long['day_of_week'] / 7)
    df_long['day_cos'] = np.cos(2 * np.pi * df_long['day_of_week'] / 7)
    df_long['month'] = df_long['Date'].dt.month
    df_long['month_sin'] = np.sin(2 * np.pi * (df_long['month'] - 1) / 12)
    df_long['month_cos'] = np.cos(2 * np.pi * (df_long['month'] - 1) / 12)
    df_long['Is_Weekend'] = df_long['day_of_week'].isin([5, 6]).astype(int)
    df_long['is_morning_rush'] = df_long['hour_of_day'].isin([8, 9, 10]).astype(int)
    df_long['is_lunch_rush'] = df_long['hour_of_day'].isin([11, 12, 13, 14]).astype(int)
    df_long['is_afternoon'] = df_long['hour_of_day'].isin([15, 16]).astype(int)
    df_long = df_long.dropna().reset_index(drop=True)
    df_long['Sales'] = df_long['Sales'].clip(lower=0)
    feature_cols = [c for c in df_long.columns if c not in ['Date', 'Sales']]
    
    if eval_mode == 'november':
        train_end = pd.to_datetime('2025-10-01')
        val_end = pd.to_datetime('2025-11-01')
        X_train = df_long[df_long['Date'] <= train_end][feature_cols]
        y_train = df_long[df_long['Date'] <= train_end]['Sales']
        X_val = df_long[(df_long['Date'] > train_end) & (df_long['Date'] <= val_end)][feature_cols]
        y_val = df_long[(df_long['Date'] > train_end) & (df_long['Date'] <= val_end)]['Sales']
        print(f"    Evaluating November 2025 (Improved Hourly, Train <= {train_end}, Val <= {val_end})...")
    else:
        max_date = df_long['Date'].max()
        val_start = max_date - pd.Timedelta(days=30)
        X_train = df_long[df_long['Date'] <= val_start][feature_cols]
        y_train = df_long[df_long['Date'] <= val_start]['Sales']
        X_val = df_long[df_long['Date'] > val_start][feature_cols]
        y_val = df_long[df_long['Date'] > val_start]['Sales']

    print(f"    Tuning Improved Hourly XGBoost ({OPTUNA_TRIALS} trials)...")
    def objective(trial):
        params = {"n_estimators": 2000, "early_stopping_rounds": 50, "learning_rate": trial.suggest_float("learning_rate", 3e-3, 0.15, log=True),
                  "max_depth": trial.suggest_int("max_depth", 4, 10), "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
                  "subsample": trial.suggest_float("subsample", 0.6, 0.95), "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
                  "gamma": trial.suggest_float("gamma", 1e-4, 2.0, log=True), "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
                  "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True), "enable_categorical": True, "tree_method": "hist"}
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return np.sqrt(np.mean((y_val - model.predict(X_val)) ** 2))
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    best_params = study.best_params
    best_params.update({"n_estimators": 2000, "early_stopping_rounds": 30, "enable_categorical": True, "tree_method": "hist"})
    best_params.pop('early_stopping_rounds', None)
    print(f"    Best RMSE: {study.best_value:.4f}. Training + forecasting...")
    
    if eval_mode == 'november':
        val_end = pd.to_datetime('2025-11-01')
        full_df = df_long[df_long['Date'] <= val_end]
        model = xgb.XGBRegressor(**best_params)
        model.fit(full_df[feature_cols], full_df['Sales'], verbose=False)
        forecast_start_df = full_df
    else:
        model = xgb.XGBRegressor(**best_params)
        model.fit(df_long[feature_cols], df_long['Sales'], verbose=False)
        forecast_start_df = df_long

    return _xgb_hourly_recursive(forecast_start_df, model, feature_cols, forecast_days,
        lag_map={1: 'sales_1h_ago', 2: 'sales_2h_ago', HOURS_PER_DAY: 'sales_same_hour_yesterday',
                 HOURS_PER_DAY * 7: 'sales_same_hour_last_week', HOURS_PER_DAY * 14: 'sales_same_hour_2weeks_ago'},
        rolling_windows=[(HOURS_PER_DAY, '1d'), (HOURS_PER_DAY * 3, '3d'), (HOURS_PER_DAY * 7, '7d')])

def _xgb_hourly_recursive(df_long, model, feature_cols, forecast_days, lag_map, rolling_windows):
    # Shared hourly XGBoost recursive forecast → aggregate to daily.
    last_datetime = df_long['Date'].max()
    last_date = last_datetime.normalize()
    # Build future hourly timestamps
    forecast_hours = []
    for day_offset in range(forecast_days):
        day = last_date + pd.Timedelta(days=day_offset + 1)
        for hour in BUSINESS_HOURS:
            forecast_hours.append(day + pd.Timedelta(hours=hour))

    products = sorted(df_long['Product_Name'].cat.categories)
    # Build sales history
    sales_history = {(row['Product_Name'], row['Date']): row['Sales'] for _, row in df_long[['Date', 'Product_Name', 'Sales']].iterrows()}

    forecast_rows = [{'Date': dt, 'Product_Name': p} for dt in forecast_hours for p in products]
    forecast_df = pd.DataFrame(forecast_rows)
    forecast_df['Product_Name'] = forecast_df['Product_Name'].astype(pd.CategoricalDtype(categories=df_long['Product_Name'].cat.categories))

    # Time features
    for col in feature_cols:
        if col not in forecast_df.columns:
            if col == 'hour_of_day':
                forecast_df[col] = forecast_df['Date'].dt.hour
            elif col == 'hour_sin':
                forecast_df[col] = np.sin(2 * np.pi * (forecast_df['Date'].dt.hour - 8) / 9)
            elif col == 'hour_cos':
                forecast_df[col] = np.cos(2 * np.pi * (forecast_df['Date'].dt.hour - 8) / 9)
            elif col == 'day_of_week':
                forecast_df[col] = forecast_df['Date'].dt.dayofweek
            elif col == 'day_sin':
                forecast_df[col] = np.sin(2 * np.pi * forecast_df['Date'].dt.dayofweek / 7)
            elif col == 'day_cos':
                forecast_df[col] = np.cos(2 * np.pi * forecast_df['Date'].dt.dayofweek / 7)
            elif col == 'month':
                forecast_df[col] = forecast_df['Date'].dt.month
            elif col == 'month_sin':
                forecast_df[col] = np.sin(2 * np.pi * (forecast_df['Date'].dt.month - 1) / 12)
            elif col == 'month_cos':
                forecast_df[col] = np.cos(2 * np.pi * (forecast_df['Date'].dt.month - 1) / 12)
            elif col == 'Is_Weekend':
                forecast_df[col] = forecast_df['Date'].dt.dayofweek.isin([5, 6]).astype(int)
            elif col == 'is_morning_rush':
                forecast_df[col] = forecast_df['Date'].dt.hour.isin([8, 9, 10]).astype(int)
            elif col == 'is_lunch_rush':
                forecast_df[col] = forecast_df['Date'].dt.hour.isin([11, 12, 13, 14]).astype(int)
            elif col == 'is_afternoon':
                forecast_df[col] = forecast_df['Date'].dt.hour.isin([15, 16]).astype(int)
            else:
                forecast_df[col] = 0

    # Init lags from history
    for row_index, row in forecast_df.iterrows():
        product_name, current_datetime = row['Product_Name'], row['Date']
        for lag_steps, lag_col in lag_map.items():
            past_datetime = current_datetime - pd.Timedelta(hours=lag_steps)
            forecast_df.at[row_index, lag_col] = sales_history.get((product_name, past_datetime), 0)
        for window_size, window_name in rolling_windows:
            window_values = [sales_history.get((product_name, current_datetime - pd.Timedelta(hours=i)), 0) for i in range(1, window_size + 1)]
            forecast_df.at[row_index, f'rolling_{window_name}_avg'] = np.mean(window_values) if window_values else 0
            forecast_df.at[row_index, f'rolling_{window_name}_std'] = np.std(window_values, ddof=1) if len(window_values) > 1 else 0
    for col in forecast_df.select_dtypes(include=[np.number]).columns:
        forecast_df[col] = forecast_df[col].fillna(0)

    index_lookup = {(row['Product_Name'], row['Date']): i for i, row in forecast_df.iterrows()}

    # Predict hour by hour
    for hour_index, current_datetime in enumerate(forecast_hours):
        hour_indices = forecast_df.index[forecast_df['Date'] == current_datetime].tolist()
        predictions = np.clip(model.predict(forecast_df.loc[hour_indices, feature_cols]), 0, None)
        forecast_df.loc[hour_indices, 'Forecast'] = predictions
        for row_index, pred_value in zip(hour_indices, predictions):
            product = forecast_df.at[row_index, 'Product_Name']
            sales_history[(product, current_datetime)] = pred_value
            for lag_steps, lag_col in lag_map.items():
                future_key = (product, current_datetime + pd.Timedelta(hours=lag_steps))
                if future_key in index_lookup:
                    forecast_df.at[index_lookup[future_key], lag_col] = pred_value
        # Update rolling for next hour
        if hour_index + 1 < len(forecast_hours):
            next_datetime = forecast_hours[hour_index + 1]
            for next_row_index in forecast_df.index[forecast_df['Date'] == next_datetime]:
                product = forecast_df.at[next_row_index, 'Product_Name']
                for window_size, window_name in rolling_windows:
                    window_values = [sales_history.get((product, next_datetime - pd.Timedelta(hours=i)), 0) for i in range(1, window_size + 1)]
                    forecast_df.at[next_row_index, f'rolling_{window_name}_avg'] = np.mean(window_values) if window_values else 0
                    forecast_df.at[next_row_index, f'rolling_{window_name}_std'] = np.std(window_values, ddof=1) if len(window_values) > 1 else 0

    # Aggregate to daily
    forecast_df['Date_Day'] = forecast_df['Date'].dt.normalize()
    daily_forecast = forecast_df.groupby(['Date_Day', 'Product_Name'])['Forecast'].sum().reset_index()
    daily_forecast = daily_forecast.rename(columns={'Date_Day': 'Date'})
    daily_forecast['Forecast'] = daily_forecast['Forecast'].round().astype(int)
    return daily_forecast[['Date', 'Product_Name', 'Forecast']]

# ── 8. LSTM GLOBAL DAILY (CPU-compatible) ──
def run_lstm_daily(forecast_days, eval_mode=None):
    #Global LSTM daily: one model for all products, 30-day sequences, recursive forecast.
    # Since not all system can eun on a GPU, this version of LSTM is CPU formated, but as a WARNING : it will take time
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.losses import Huber
    from tensorflow.keras.regularizers import l2
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    import optuna; optuna.logging.set_verbosity(logging.WARNING)

    SEQUENCE_LENGTH = 30
    OPTUNA_LSTM_TRIALS = 10  # Fewer trials for CPU speed

    print("    Loading data for LSTM Daily (CPU mode)...")
    df_long, daily_sales = load_sales_long()
    daily_weather, daily_holidays, daily_events = load_exogenous()

    # Build wide format with time + exogenous features
    product_cols = [col for col in daily_sales.columns if col != 'Date']
    daily_sales['day_sin'] = np.sin(2 * np.pi * daily_sales['Date'].dt.dayofweek / 7)
    daily_sales['day_cos'] = np.cos(2 * np.pi * daily_sales['Date'].dt.dayofweek / 7)
    daily_sales['month_sin'] = np.sin(2 * np.pi * (daily_sales['Date'].dt.month - 1) / 12)
    daily_sales['month_cos'] = np.cos(2 * np.pi * (daily_sales['Date'].dt.month - 1) / 12)
    daily_sales['Is_Weekend'] = daily_sales['Date'].dt.dayofweek.isin([5, 6]).astype(int)
    time_features = ['day_sin', 'day_cos', 'month_sin', 'month_cos', 'Is_Weekend']

    exclude_cols = ['Date', 'Time', 'date']
    weather_features = [col for col in daily_weather.columns if col not in exclude_cols]
    holiday_features = [col for col in daily_holidays.columns if col not in exclude_cols]
    event_features = [col for col in daily_events.columns if col not in exclude_cols]

    df_wide = daily_sales.merge(daily_weather, on='Date', how='left')
    df_wide = df_wide.merge(daily_holidays, on='Date', how='left')
    df_wide = df_wide.merge(daily_events, on='Date', how='left')
    for col in df_wide.select_dtypes(include=[np.number]).columns:
        df_wide[col] = df_wide[col].fillna(0)

    base_feature_cols = [col for col in weather_features + holiday_features + event_features + time_features
                         if col in df_wide.columns and df_wide[col].dtype.kind in 'iufc']

    # Melt to long
    df_long_format = pd.melt(df_wide, id_vars=['Date'] + base_feature_cols, value_vars=product_cols,
                             var_name='Product_Name', value_name='Sales')
    df_long_format['Sales'] = df_long_format['Sales'].clip(lower=0)
    df_long_format = df_long_format[df_long_format['Product_Name'].isin(PRODUCTS_TO_FORECAST)]
    df_long_format = df_long_format.sort_values(['Product_Name', 'Date']).reset_index(drop=True)

    product_encoder = LabelEncoder()
    df_long_format['product_id'] = product_encoder.fit_transform(df_long_format['Product_Name'])

    # Lag features
    df_long_format['sales_lag_1'] = df_long_format.groupby('Product_Name')['Sales'].shift(1)
    df_long_format['sales_lag_7'] = df_long_format.groupby('Product_Name')['Sales'].shift(7)
    df_long_format['sales_rolling_7_mean'] = df_long_format.groupby('Product_Name')['Sales'].shift(1).groupby(
        df_long_format['Product_Name']).transform(lambda x: x.rolling(7, min_periods=1).mean())
    df_long_format['sales_rolling_7_std'] = df_long_format.groupby('Product_Name')['Sales'].shift(1).groupby(
        df_long_format['Product_Name']).transform(lambda x: x.rolling(7, min_periods=1).std()).fillna(0)
    df_long_format['sales_diff_1'] = df_long_format.groupby('Product_Name')['Sales'].diff(1)

    lag_features = ['sales_lag_1', 'sales_lag_7', 'sales_rolling_7_mean', 'sales_rolling_7_std', 'sales_diff_1']
    df_long_format = df_long_format.dropna(subset=lag_features).reset_index(drop=True)
    final_feature_cols = base_feature_cols + ['product_id'] + lag_features

    # Split
    if eval_mode == 'november':
        train_end = pd.to_datetime('2025-10-01')
        val_end = pd.to_datetime('2025-11-01')
        train_data = df_long_format[df_long_format['Date'] <= train_end]
        val_data = df_long_format[(df_long_format['Date'] > train_end) & (df_long_format['Date'] <= val_end)]
        print(f"    Evaluating November 2025 (LSTM Daily, Train <= {train_end}, Val <= {val_end})...")
    else:
        max_date = df_long_format['Date'].max()
        train_end = max_date - pd.Timedelta(days=60)
        val_end = max_date - pd.Timedelta(days=30)
        train_data = df_long_format[df_long_format['Date'] <= train_end]
        val_data = df_long_format[(df_long_format['Date'] > train_end) & (df_long_format['Date'] <= val_end)]

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feature_scaler.fit(train_data[final_feature_cols])
    target_scaler.fit(train_data[['Sales']])

    # Build sequences
    def build_sequences(pdf):
        if len(pdf) < SEQUENCE_LENGTH + 1:
            return None, None, None
        scaled_features = feature_scaler.transform(pdf[final_feature_cols])
        scaled_target = target_scaler.transform(pdf[['Sales']]).flatten()
        X, y, dates = [], [], []
        for i in range(len(scaled_features) - SEQUENCE_LENGTH):
            X.append(scaled_features[i : i + SEQUENCE_LENGTH])
            y.append(scaled_target[i + SEQUENCE_LENGTH])
            dates.append(pdf['Date'].values[i + SEQUENCE_LENGTH])
        return np.array(X), np.array(y), pd.to_datetime(dates)

    all_X_train, all_y_train, all_X_val, all_y_val = [], [], [], []
    for product_name in df_long_format['Product_Name'].unique():
        product_df = df_long_format[df_long_format['Product_Name'] == product_name].sort_values('Date')
        res = build_sequences(product_df)
        if res[0] is None:
            continue
        X_seqs, y_seqs, d_seqs = res
        train_mask = d_seqs <= np.datetime64(train_end)
        val_mask = (d_seqs > np.datetime64(train_end)) & (d_seqs <= np.datetime64(val_end))
        if train_mask.sum() > 0:
            all_X_train.append(X_seqs[train_mask])
            all_y_train.append(y_seqs[train_mask])
        if val_mask.sum() > 0:
            all_X_val.append(X_seqs[val_mask])
            all_y_val.append(y_seqs[val_mask])

    X_train = np.vstack(all_X_train)
    y_train = np.concatenate(all_y_train)
    X_val = np.vstack(all_X_val)
    y_val = np.concatenate(all_y_val)
    shuffle_indices = np.random.permutation(len(X_train))
    X_train, y_train = X_train[shuffle_indices], y_train[shuffle_indices]

    print(f"    Sequences: {X_train.shape[0]} train, {X_val.shape[0]} val")
    print(f"    Tuning LSTM ({OPTUNA_LSTM_TRIALS} trials, CPU)...")

    def objective(trial):
        tf.keras.backend.clear_session()
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
        lstm_1_units = trial.suggest_int('lstm_1_units', 64, 192, step=32)
        lstm_2_units = trial.suggest_int('lstm_2_units', 32, 96, step=16)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)
        model = Sequential([Input(shape=(SEQUENCE_LENGTH, len(final_feature_cols))),
                            LSTM(lstm_1_units, activation='tanh', return_sequences=True), Dropout(dropout_rate),
                            LSTM(lstm_2_units, activation='tanh'), Dropout(dropout_rate),
                            Dense(64, activation='relu', kernel_regularizer=l2(l2_reg)),
                            Dense(32, activation='relu'), Dense(1)])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=Huber())
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=batch_size,
                  callbacks=[EarlyStopping(patience=8, restore_best_weights=True)], verbose=0)
        val_predictions = target_scaler.inverse_transform(model.predict(X_val, verbose=0)).flatten()
        val_actuals = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
        return np.sqrt(np.mean((val_actuals - val_predictions) ** 2))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=OPTUNA_LSTM_TRIALS)
    best_params = study.best_params
    print(f"    Best RMSE: {study.best_value:.4f}. Training final model...")

    tf.keras.backend.clear_session()
    final_model = Sequential([Input(shape=(SEQUENCE_LENGTH, len(final_feature_cols))),
                              LSTM(best_params['lstm_1_units'], activation='tanh', return_sequences=True), Dropout(best_params['dropout_rate']),
                              LSTM(best_params['lstm_2_units'], activation='tanh'), Dropout(best_params['dropout_rate']),
                              Dense(64, activation='relu', kernel_regularizer=l2(best_params['l2_reg'])),
                              Dense(32, activation='relu'), Dense(1)])
    final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss=Huber())
    final_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=150,
                    batch_size=best_params['batch_size'],
                    callbacks=[EarlyStopping(patience=15, restore_best_weights=True),
                               ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)], verbose=0)

    # Recursive forecast
    print(f"    Forecasting {forecast_days} days recursively...")
    lag_indices = [final_feature_cols.index(f) for f in lag_features]
    lag_mins = feature_scaler.data_min_[lag_indices]
    lag_ranges = feature_scaler.data_range_[lag_indices]
    lag_ranges[lag_ranges == 0] = 1.0

    forecast_results = []
    for product_name in sorted(df_long_format['Product_Name'].unique()):
        product_df = df_long_format[df_long_format['Product_Name'] == product_name].sort_values('Date')
        if len(product_df) < SEQUENCE_LENGTH:
            continue
        product_tail = product_df.tail(SEQUENCE_LENGTH)
        current_sequence = feature_scaler.transform(product_tail[final_feature_cols])
        sales_history_list = list(product_df['Sales'].values[-SEQUENCE_LENGTH:])

        for day_idx in range(forecast_days):
            prediction_scaled = final_model.predict(current_sequence.reshape(1, SEQUENCE_LENGTH, -1), verbose=0)[0, 0]
            prediction_raw = max(0, target_scaler.inverse_transform([[prediction_scaled]])[0, 0])
            sales_history_list.append(prediction_raw)
            forecast_date = max_date + pd.Timedelta(days=day_idx + 1)
            forecast_results.append({'Date': forecast_date, 'Product_Name': product_name, 'Forecast': int(round(prediction_raw))})

            if day_idx + 1 < forecast_days:
                next_features = current_sequence[-1].copy()
                # Update time features
                next_date = forecast_date + pd.Timedelta(days=1)
                day_of_week = next_date.dayofweek
                month = next_date.month
                time_values = [np.sin(2 * np.pi * day_of_week / 7), np.cos(2 * np.pi * day_of_week / 7),
                               np.sin(2 * np.pi * (month - 1) / 12), np.cos(2 * np.pi * (month - 1) / 12),
                               1 if day_of_week >= 5 else 0]
                for ti, tf_name in enumerate(time_features):
                    if tf_name in final_feature_cols:
                        fi = final_feature_cols.index(tf_name)
                        next_features[fi] = (time_values[ti] - feature_scaler.data_min_[fi]) / (feature_scaler.data_range_[fi] + 1e-8)
                # Update lags from sales_history_list
                history = sales_history_list
                raw_lags = np.array([history[-1], history[-7] if len(history) >= 7 else history[0],
                                     np.mean(history[-7:]) if len(history) >= 7 else np.mean(history),
                                     np.std(history[-7:]) if len(history) >= 7 else 0.0,
                                     history[-1] - history[-2] if len(history) >= 2 else 0.0])
                scaled_lags = (raw_lags - lag_mins) / lag_ranges
                for li, fi in enumerate(lag_indices):
                    next_features[fi] = scaled_lags[li]
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1] = next_features

    tf.keras.backend.clear_session()
    return pd.DataFrame(forecast_results)


# ── 9. LSTM GLOBAL HOURLY (CPU-compatible) ──
def run_lstm_hourly(forecast_days, eval_mode=None):
    #Global LSTM hourly: one model for all products, 63-step sequences, 9h→daily rollup.
    # Since not all system can eun on a GPU, this version of LSTM is CPU formated, but as a WARNING : it will take time
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.losses import Huber
    from tensorflow.keras.regularizers import l2
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    import optuna; optuna.logging.set_verbosity(logging.WARNING)

    SEQUENCE_LENGTH = 63  # ~1 week of hourly data (9h × 7d)
    OPTUNA_LSTM_TRIALS = 8

    print("    Loading hourly data for LSTM Hourly (CPU mode)...")
    df_long_hourly = load_sales_hourly()
    daily_weather, daily_holidays, daily_events = load_exogenous()

    # Time features on hourly data
    df_long_hourly['day_sin'] = np.sin(2 * np.pi * df_long_hourly['Date'].dt.dayofweek / 7)
    df_long_hourly['day_cos'] = np.cos(2 * np.pi * df_long_hourly['Date'].dt.dayofweek / 7)
    df_long_hourly['month_sin'] = np.sin(2 * np.pi * (df_long_hourly['Date'].dt.month - 1) / 12)
    df_long_hourly['month_cos'] = np.cos(2 * np.pi * (df_long_hourly['Date'].dt.month - 1) / 12)
    df_long_hourly['Is_Weekend'] = df_long_hourly['Date'].dt.dayofweek.isin([5, 6]).astype(int)
    df_long_hourly['hour_sin'] = np.sin(2 * np.pi * (df_long_hourly['Date'].dt.hour - 8) / 9)
    df_long_hourly['hour_cos'] = np.cos(2 * np.pi * (df_long_hourly['Date'].dt.hour - 8) / 9)
    time_features = ['day_sin', 'day_cos', 'month_sin', 'month_cos', 'Is_Weekend', 'hour_sin', 'hour_cos']

    # Merge daily exogenous (broadcast to each hour of that day)
    df_long_hourly['Date_Day'] = df_long_hourly['Date'].dt.normalize()
    df_long_hourly = df_long_hourly.merge(daily_weather.rename(columns={'Date': 'Date_Day'}), on='Date_Day', how='left')
    df_long_hourly = df_long_hourly.merge(daily_holidays.rename(columns={'Date': 'Date_Day'}), on='Date_Day', how='left')
    df_long_hourly = df_long_hourly.merge(daily_events.rename(columns={'Date': 'Date_Day'}), on='Date_Day', how='left')
    df_long_hourly = df_long_hourly.drop(columns=['Date_Day'])
    for col in df_long_hourly.select_dtypes(include=[np.number]).columns:
        df_long_hourly[col] = df_long_hourly[col].fillna(0)

    exclude_cols = ['Date', 'Sales', 'Product_Name', 'Time', 'date']
    exogenous_cols = [col for col in daily_weather.columns.tolist() + daily_holidays.columns.tolist() + daily_events.columns.tolist()
                      if col not in ['Date', 'Time', 'date'] and col in df_long_hourly.columns]
    # Deduplicate preserving order
    exogenous_cols = list(dict.fromkeys(exogenous_cols))
    exogenous_cols = [col for col in exogenous_cols if df_long_hourly[col].dtype.kind in 'iufc']

    df_long_hourly = df_long_hourly.sort_values(['Product_Name', 'Date']).reset_index(drop=True)
    product_encoder = LabelEncoder()
    df_long_hourly['Product_ID'] = product_encoder.fit_transform(df_long_hourly['Product_Name'])

    # Hourly lags
    df_long_hourly['sales_lag_1h'] = df_long_hourly.groupby('Product_Name')['Sales'].shift(1)
    df_long_hourly['sales_lag_9h'] = df_long_hourly.groupby('Product_Name')['Sales'].shift(9)
    df_long_hourly['sales_lag_63h'] = df_long_hourly.groupby('Product_Name')['Sales'].shift(63)
    df_long_hourly['sales_rolling_9h_mean'] = df_long_hourly.groupby('Product_Name')['Sales'].shift(1).groupby(
        df_long_hourly['Product_Name']).transform(lambda x: x.rolling(9, min_periods=1).mean())
    df_long_hourly['sales_rolling_9h_std'] = df_long_hourly.groupby('Product_Name')['Sales'].shift(1).groupby(
        df_long_hourly['Product_Name']).transform(lambda x: x.rolling(9, min_periods=1).std()).fillna(0)

    lag_features = ['sales_lag_1h', 'sales_lag_9h', 'sales_lag_63h', 'sales_rolling_9h_mean', 'sales_rolling_9h_std']
    df_long_hourly = df_long_hourly.dropna(subset=lag_features).reset_index(drop=True)
    feature_cols = exogenous_cols + time_features + ['Product_ID'] + lag_features
    # Deduplicate
    feature_cols = list(dict.fromkeys(feature_cols))
    feature_cols = [col for col in feature_cols if col in df_long_hourly.columns]

    if eval_mode == 'november':
        train_end = pd.to_datetime('2025-10-01')
        val_end = pd.to_datetime('2025-11-01')
        print(f"    Evaluating November 2025 (LSTM Hourly, Train <= {train_end}, Val <= {val_end})...")
    else:
        max_date = df_long_hourly['Date'].max()
        train_end = max_date - pd.Timedelta(days=60)
        val_end = max_date - pd.Timedelta(days=30)

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    train_data = df_long_hourly[df_long_hourly['Date'] <= train_end]
    feature_scaler.fit(train_data[feature_cols])
    target_scaler.fit(train_data[['Sales']])

    def build_sequences(pdf):
        if len(pdf) < SEQUENCE_LENGTH + 1:
            return None, None, None
        scaled_features = feature_scaler.transform(pdf[feature_cols])
        scaled_target = target_scaler.transform(pdf[['Sales']]).flatten()
        X, y, dates = [], [], []
        for i in range(len(scaled_features) - SEQUENCE_LENGTH):
            X.append(scaled_features[i : i + SEQUENCE_LENGTH])
            y.append(scaled_target[i + SEQUENCE_LENGTH])
            dates.append(pdf['Date'].values[i + SEQUENCE_LENGTH])
        return np.array(X), np.array(y), pd.to_datetime(dates)

    all_X_train, all_y_train, all_X_val, all_y_val = [], [], [], []
    for product_name in df_long_hourly['Product_Name'].unique():
        product_df = df_long_hourly[df_long_hourly['Product_Name'] == product_name].sort_values('Date')
        res = build_sequences(product_df)
        if res[0] is None:
            continue
        X_seqs, y_seqs, d_seqs = res
        train_mask = d_seqs <= np.datetime64(train_end)
        val_mask = (d_seqs > np.datetime64(train_end)) & (d_seqs <= np.datetime64(val_end))
        if train_mask.sum() > 0:
            all_X_train.append(X_seqs[train_mask])
            all_y_train.append(y_seqs[train_mask])
        if val_mask.sum() > 0:
            all_X_val.append(X_seqs[val_mask])
            all_y_val.append(y_seqs[val_mask])

    X_train = np.vstack(all_X_train)
    y_train = np.concatenate(all_y_train)
    X_val = np.vstack(all_X_val)
    y_val = np.concatenate(all_y_val)
    shuffle_indices = np.random.permutation(len(X_train))
    X_train, y_train = X_train[shuffle_indices], y_train[shuffle_indices]

    print(f"    Sequences: {X_train.shape[0]} train, {X_val.shape[0]} val")
    print(f"    Tuning LSTM Hourly ({OPTUNA_LSTM_TRIALS} trials, CPU)...")

    def objective(trial):
        tf.keras.backend.clear_session()
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
        lstm_1_units = trial.suggest_int('lstm_1_units', 64, 192, step=32)
        lstm_2_units = trial.suggest_int('lstm_2_units', 32, 96, step=16)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)
        model = Sequential([Input(shape=(SEQUENCE_LENGTH, len(feature_cols))),
                            LSTM(lstm_1_units, activation='tanh', return_sequences=True), Dropout(dropout_rate),
                            LSTM(lstm_2_units, activation='tanh'), Dropout(dropout_rate),
                            Dense(64, activation='relu', kernel_regularizer=l2(l2_reg)),
                            Dense(1)])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=Huber())
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=batch_size,
                  callbacks=[EarlyStopping(patience=6, restore_best_weights=True)], verbose=0)
        val_predictions = target_scaler.inverse_transform(model.predict(X_val, verbose=0)).flatten()
        val_actuals = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
        return np.sqrt(np.mean((val_actuals - val_predictions) ** 2))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=OPTUNA_LSTM_TRIALS)
    best_params = study.best_params
    print(f"    Best RMSE: {study.best_value:.4f}. Training final model...")

    tf.keras.backend.clear_session()
    final_model = Sequential([Input(shape=(SEQUENCE_LENGTH, len(feature_cols))),
                              LSTM(best_params['lstm_1_units'], activation='tanh', return_sequences=True), Dropout(best_params['dropout_rate']),
                              LSTM(best_params['lstm_2_units'], activation='tanh'), Dropout(best_params['dropout_rate']),
                              Dense(64, activation='relu', kernel_regularizer=l2(best_params['l2_reg'])),
                              Dense(1)])
    final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss=Huber())
    final_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100,
                    batch_size=best_params['batch_size'],
                    callbacks=[EarlyStopping(patience=12, restore_best_weights=True),
                               ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)], verbose=0)

    # Recursive hourly forecast → aggregate to daily
    print(f"    Forecasting {forecast_days} days recursively (hourly)...")
    lag_indices = [feature_cols.index(f) for f in lag_features]
    lag_mins = feature_scaler.data_min_[lag_indices]
    lag_ranges = feature_scaler.data_range_[lag_indices]
    lag_ranges[lag_ranges == 0] = 1.0

    forecast_results = []
    for product_name in sorted(df_long_hourly['Product_Name'].unique()):
        product_df = df_long_hourly[df_long_hourly['Product_Name'] == product_name].sort_values('Date')
        if len(product_df) < SEQUENCE_LENGTH:
            continue
        product_tail = product_df.tail(SEQUENCE_LENGTH)
        current_sequence = feature_scaler.transform(product_tail[feature_cols])
        sales_history_list = list(product_df['Sales'].values[-max(SEQUENCE_LENGTH, 63) :])
        max_dt = product_df['Date'].max()

        forecast_hours_list = []
        for day_offset in range(forecast_days):
            day_dt = max_dt.normalize() + pd.Timedelta(days=day_offset + 1)
            for hour in BUSINESS_HOURS:
                forecast_hours_list.append(day_dt + pd.Timedelta(hours=hour))

        for hour_idx, current_datetime in enumerate(forecast_hours_list):
            prediction_scaled = final_model.predict(current_sequence.reshape(1, SEQUENCE_LENGTH, -1), verbose=0)[0, 0]
            prediction_raw = max(0, target_scaler.inverse_transform([[prediction_scaled]])[0, 0])
            sales_history_list.append(prediction_raw)
            forecast_results.append({'Date': current_datetime, 'Product_Name': product_name, 'Forecast_h': prediction_raw})

            if hour_idx + 1 < len(forecast_hours_list):
                next_features = current_sequence[-1].copy()
                # Recompute hourly lags
                history = sales_history_list
                raw_lags = np.array([history[-1], history[-9] if len(history) >= 9 else history[0],
                                     history[-63] if len(history) >= 63 else history[0],
                                     np.mean(history[-9:]) if len(history) >= 9 else np.mean(history),
                                     np.std(history[-9:]) if len(history) >= 9 else 0.0])
                scaled_lags = (raw_lags - lag_mins) / lag_ranges
                for li, fi in enumerate(lag_indices):
                    next_features[fi] = scaled_lags[li]
                # Update time features for next hour
                next_datetime = forecast_hours_list[hour_idx + 1]
                day_of_week = next_datetime.dayofweek
                month = next_datetime.month
                hour = next_datetime.hour
                time_values = {'day_sin': np.sin(2 * np.pi * day_of_week / 7), 'day_cos': np.cos(2 * np.pi * day_of_week / 7),
                               'month_sin': np.sin(2 * np.pi * (month - 1) / 12), 'month_cos': np.cos(2 * np.pi * (month - 1) / 12),
                               'Is_Weekend': 1 if day_of_week >= 5 else 0,
                               'hour_sin': np.sin(2 * np.pi * (hour - 8) / 9), 'hour_cos': np.cos(2 * np.pi * (hour - 8) / 9)}
                for tf_name, tv in time_values.items():
                    if tf_name in feature_cols:
                        fi = feature_cols.index(tf_name)
                        next_features[fi] = (tv - feature_scaler.data_min_[fi]) / (feature_scaler.data_range_[fi] + 1e-8)
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1] = next_features

    tf.keras.backend.clear_session()

    # Rollup hourly → daily
    rollup_df = pd.DataFrame(forecast_results)
    rollup_df['Date_Day'] = rollup_df['Date'].dt.normalize()
    daily_forecast = rollup_df.groupby(['Date_Day', 'Product_Name'])['Forecast_h'].sum().reset_index()
    daily_forecast = daily_forecast.rename(columns={'Date_Day': 'Date', 'Forecast_h': 'Forecast'})
    daily_forecast['Forecast'] = daily_forecast['Forecast'].round().astype(int)
    return daily_forecast[['Date', 'Product_Name', 'Forecast']]


# ── RUNNER REGISTRY ──
MODEL_RUNNERS = {
    'xgb_simple_daily':   run_xgb_simple_daily,
    'xgb_improved_daily': run_xgb_improved_daily,
    'xgb_simple_hourly':  run_xgb_simple_hourly,
    'xgb_improved_hourly':run_xgb_improved_hourly,
    'arima_forecast':              run_arima,
    'prophet_daily':      run_prophet_daily,
    'prophet_hourly':     run_prophet_hourly,
    'lstm_daily':         run_lstm_daily,
    'lstm_hourly':        run_lstm_hourly,
}


# ══════════════════════════════════════════════════════════════
# PDF REPORT (generate_report function)
# ══════════════════════════════════════════════════════════════


def generate_report(forecast_df, daily_ingredients, model_name, model_info, horizon, output_path):
    """Generate PDF report — identical to v2 implementation."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.enums import TA_CENTER

    doc = SimpleDocTemplate(output_path, pagesize=A4, topMargin=1.5*cm, bottomMargin=1.5*cm,
                            leftMargin=1.5*cm, rightMargin=1.5*cm)
    styles = getSampleStyleSheet()
    W = A4[0] - 3*cm
    title_s=ParagraphStyle('T',parent=styles['Title'],fontSize=22,spaceAfter=6,textColor=colors.HexColor('#1a1a2e'))
    sub_s=ParagraphStyle('S',parent=styles['Normal'],fontSize=12,textColor=colors.HexColor('#555'),spaceAfter=10,alignment=TA_CENTER)
    sec_s=ParagraphStyle('Sec',parent=styles['Heading2'],fontSize=14,textColor=colors.HexColor('#16213e'),spaceBefore=10,spaceAfter=6)
    subsec_s=ParagraphStyle('Sub',parent=styles['Heading3'],fontSize=11,textColor=colors.HexColor('#0f3460'),spaceBefore=8,spaceAfter=4)
    body_s=ParagraphStyle('B',parent=styles['Normal'],fontSize=9,leading=12)
    note_s=ParagraphStyle('N',parent=styles['Normal'],fontSize=8,textColor=colors.HexColor('#666'),spaceAfter=4,leading=11)
    hdr_bg=colors.HexColor('#1a1a2e'); hdr_fg=colors.white; alt=colors.HexColor('#f0f0f5')
    grid=colors.HexColor('#cccccc'); ingr_hdr=colors.HexColor('#0f3460'); total_bg=colors.HexColor('#e8e8f0')
    story=[]; forecast_dates=sorted(forecast_df['Date'].unique()); n_days=len(forecast_dates)
    total_items=int(forecast_df['Forecast'].sum())
    DAY_NAMES=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

    def make_table(data, col_widths, header_color=hdr_bg):
        table = Table(data, colWidths=col_widths, repeatRows=1)
        table_style = [('BACKGROUND', (0, 0), (-1, 0), header_color), ('TEXTCOLOR', (0, 0), (-1, 0), hdr_fg),
                       ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, 0), 8),
                       ('FONTSIZE', (0, 1), (-1, -1), 8), ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                       ('ALIGN', (0, 0), (0, -1), 'LEFT'), ('GRID', (0, 0), (-1, -1), 0.3, grid),
                       ('BOTTOMPADDING', (0, 0), (-1, -1), 3), ('TOPPADDING', (0, 0), (-1, -1), 3)]
        for i in range(1, len(data)):
            if i % 2 == 0:
                table_style.append(('BACKGROUND', (0, i), (-1, i), alt))
        table.setStyle(TableStyle(table_style))
        return table

    def add_two_col(items, col_label1, col_label2, header_color=ingr_hdr):
        half_length = (len(items) + 1) // 2
        left_items, right_items = items[:half_length], items[half_length:]
        data = [[col_label1, col_label2, '', col_label1, col_label2]]
        for i in range(half_length):
            left_name = left_items[i][0] if i < len(left_items) else ''
            left_value = left_items[i][1] if i < len(left_items) else ''
            right_name = right_items[i][0] if i < len(right_items) else ''
            right_value = right_items[i][1] if i < len(right_items) else ''
            data.append([left_name, left_value, '', right_name, right_value])
        table = Table(data, colWidths=[4 * cm, 2.5 * cm, 0.4 * cm, 4 * cm, 2.5 * cm])
        table_style = [('BACKGROUND', (0, 0), (1, 0), header_color), ('BACKGROUND', (3, 0), (4, 0), header_color),
                       ('TEXTCOLOR', (0, 0), (-1, 0), hdr_fg), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                       ('FONTSIZE', (0, 0), (-1, -1), 8), ('ALIGN', (1, 0), (1, -1), 'RIGHT'), ('ALIGN', (4, 0), (4, -1), 'RIGHT'),
                       ('GRID', (0, 0), (1, -1), 0.3, grid), ('GRID', (3, 0), (4, -1), 0.3, grid),
                       ('BOTTOMPADDING', (0, 0), (-1, -1), 2), ('TOPPADDING', (0, 0), (-1, -1), 2)]
        for i in range(1, len(data)):
            if i % 2 == 0:
                table_style.append(('BACKGROUND', (0, i), (1, i), alt))
                table_style.append(('BACKGROUND', (3, i), (4, i), alt))
        table.setStyle(TableStyle(table_style))
        return table

    # Cover
    TITLES={'day':'Daily Kitchen Prep Sheet','week':'Weekly Ordering Guide','month':'Monthly Forecast Report'}
    story.append(Spacer(1,2.5*cm)); story.append(Paragraph(TITLES[horizon],title_s))
    story.append(Paragraph(f"Model: {model_name}",sub_s))
    dr=(f"{forecast_dates[0].strftime('%A %d %B %Y')}" if n_days==1
        else f"{forecast_dates[0].strftime('%d %b %Y')} — {forecast_dates[-1].strftime('%d %b %Y')}")
    story.append(Paragraph(dr,sub_s))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y at %H:%M')}",sub_s))
    story.append(Spacer(1,0.8*cm))
    if model_info and model_info.get('wape'):
        wp=f"{model_info['wape']*100:.1f}%"; ms=f"{model_info['mase']:.2f}"
        bd="under-predicting" if model_info['bias']<0 else "over-predicting"
        story.append(Paragraph(f"This model was automatically selected as the best performer for this horizon based on historical accuracy. "
            f"WAPE: {wp} (percentage of volume forecast incorrectly), MASE: {ms} ({'beating' if model_info['mase']<1 else 'not beating'} "
            f"naive baseline), Bias: {model_info['bias']:.2f} units ({bd} on average).",note_s))
        story.append(Spacer(1,0.5*cm))
    summary=[['Forecast Period',dr],['Total Items',str(total_items)],['Daily Average',str(int(total_items/n_days))],
             ['Products',str(forecast_df['Product_Name'].nunique())],['Model',model_name]]
    if model_info and model_info.get('wape'): summary.append(['Accuracy (WAPE)',f"{model_info['wape']*100:.1f}%"])
    st=Table(summary,colWidths=[5*cm,8*cm])
    st.setStyle(TableStyle([('FONTSIZE',(0,0),(-1,-1),10),('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),
        ('TEXTCOLOR',(0,0),(0,-1),colors.HexColor('#1a1a2e')),('BOTTOMPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,-2),0.5,grid)]))
    story.append(st); story.append(PageBreak())

    if horizon == 'day':
        for current_date in forecast_dates:
            day_name = DAY_NAMES[current_date.dayofweek]
            story.append(Paragraph(f"Kitchen Prep — {day_name} {current_date.strftime('%d %B %Y')}", sec_s))
            story.append(Paragraph("Forecast quantities for each menu item, sorted by expected volume. Prepare these before service.", note_s))
            story.append(Paragraph("Items to Prepare", subsec_s))
            day_predictions = forecast_df[(forecast_df['Date'] == current_date) & (forecast_df['Forecast'] > 0)].sort_values('Forecast', ascending=False)
            items_to_prep = [(str(row['Product_Name']), str(int(row['Forecast']))) for _, row in day_predictions.iterrows()]
            if items_to_prep:
                story.append(add_two_col(items_to_prep, 'Item', 'Qty', hdr_bg))
                story.append(Paragraph(f"Day total: {int(day_predictions['Forecast'].sum())} items", body_s))
            story.append(Spacer(1, 4 * mm))
            story.append(Paragraph("Ingredients Required", subsec_s))
            story.append(Paragraph("Total raw ingredients for today. Quantities over 500g shown in kg. Check stock and defrost accordingly.", note_s))
            needed_ingredients = daily_ingredients[(daily_ingredients['Date'] == current_date) & (daily_ingredients['total_qty'] > 0)]
            needed_ingredients = convert_grams_to_kg(needed_ingredients).sort_values('total_qty', ascending=False)
            ingredient_rows = [(str(row['ingredient']), fmt_qty(row['total_qty'], row['unit'])) for _, row in needed_ingredients.iterrows()]
            if ingredient_rows:
                story.append(add_two_col(ingredient_rows, 'Ingredient', 'Quantity'))
            if current_date != forecast_dates[-1]:
                story.append(PageBreak())

    elif horizon == 'week':
        story.append(Paragraph("Weekly Product Overview", sec_s))
        story.append(Paragraph("Forecast per product per day. A dash means zero. Use for daily prep planning and staffing.", note_s))
        pivot_table = forecast_df.pivot_table(index='Product_Name', columns='Date', values='Forecast', aggfunc='sum', fill_value=0)
        pivot_table['TOTAL'] = pivot_table.sum(axis=1)
        pivot_table = pivot_table[pivot_table['TOTAL'] > 0].sort_values('TOTAL', ascending=False)
        headers = ['Product'] + [f"{DAY_NAMES[d.dayofweek][:3]}\n{d.day}" for d in forecast_dates] + ['Total']
        table_data = [headers]
        for product_name, row in pivot_table.iterrows():
            row_data = [str(product_name)] + [str(int(row.get(d, 0))) if int(row.get(d, 0)) > 0 else '-' for d in forecast_dates] + [str(int(row['TOTAL']))]
            table_data.append(row_data)
        table_data.append(['TOTAL'] + [str(int(pivot_table[d].sum())) for d in forecast_dates] + [str(int(pivot_table['TOTAL'].sum()))])
        num_cols = len(headers)
        col_widths = [3.8 * cm] + [1.6 * cm] * (num_cols - 2) + [1.8 * cm]
        total_table_width = sum(col_widths)
        if total_table_width > W:
            col_widths = [w * W / total_table_width for w in col_widths]
        table = Table(table_data, colWidths=col_widths, repeatRows=1)
        table_style_list = [('BACKGROUND', (0, 0), (-1, 0), hdr_bg), ('TEXTCOLOR', (0, 0), (-1, 0), hdr_fg),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, -1), 7),
                            ('ALIGN', (1, 0), (-1, -1), 'CENTER'), ('ALIGN', (0, 0), (0, -1), 'LEFT'), ('GRID', (0, 0), (-1, -1), 0.3, grid),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 3), ('TOPPADDING', (0, 0), (-1, -1), 3),
                            ('BACKGROUND', (0, -1), (-1, -1), total_bg), ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold')]
        for i in range(1, len(table_data) - 1):
            if i % 2 == 0:
                table_style_list.append(('BACKGROUND', (0, i), (-1, i), alt))
        table.setStyle(TableStyle(table_style_list))
        story.append(table)
        story.append(PageBreak())
        story.append(Paragraph("Supplier Order List — Weekly Totals", sec_s))
        story.append(Paragraph("Total ingredients for the week, sorted by volume. Quantities over 500g shown in kg. Consider a 10-15% buffer for popular items.", note_s))
        weekly_ingredients = (daily_ingredients.groupby(['ingredient', 'unit'])['total_qty'].sum().reset_index().sort_values('total_qty', ascending=False))
        weekly_ingredients = weekly_ingredients[weekly_ingredients['total_qty'] > 0]
        weekly_ingredients = convert_grams_to_kg(weekly_ingredients)
        ingredient_rows = [(str(row['ingredient']), fmt_qty(row['total_qty'], row['unit'])) for _, row in weekly_ingredients.iterrows()]
        if ingredient_rows:
            story.append(add_two_col(ingredient_rows, 'Ingredient', 'Order Qty'))

    elif horizon == 'month':
        story.append(Paragraph("Weekly Breakdown", sec_s))
        story.append(Paragraph("Monthly forecast split by calendar week for staffing and inventory planning.", note_s))
        forecast_df = forecast_df.copy()
        forecast_df['week'] = forecast_df['Date'].dt.isocalendar().week.astype(int)
        weekly_totals = forecast_df.groupby('week')['Forecast'].sum().reset_index()
        weekly_table_data = [['Week', 'Total', 'Daily Avg']]
        for _, week_row in weekly_totals.iterrows():
            week_num = int(week_row['week'])
            total_qty = int(week_row['Forecast'])
            days_in_week = len(forecast_df[forecast_df['week'] == week_num]['Date'].unique())
            weekly_table_data.append([f"Week {week_num}", str(total_qty), str(int(total_qty / days_in_week) if days_in_week > 0 else 0)])
        story.append(make_table(weekly_table_data, [3 * cm, 4 * cm, 4 * cm]))
        story.append(Spacer(1, 6 * mm))
        story.append(Paragraph("Top 15 Products by Volume", sec_s))
        story.append(Paragraph("Highest-demand products. Ensure reliable supply and sufficient prep capacity.", note_s))
        top_products = forecast_df.groupby('Product_Name')['Forecast'].sum().sort_values(ascending=False).head(15).reset_index()
        top_table_data = [['#', 'Product', 'Monthly Total', 'Daily Avg']]
        for i, (_, row) in enumerate(top_products.iterrows(), 1):
            top_table_data.append([str(i), str(row['Product_Name']), str(int(row['Forecast'])), f"{row['Forecast']/n_days:.1f}"])
        story.append(make_table(top_table_data, [1 * cm, 5 * cm, 3 * cm, 3 * cm]))
        story.append(Spacer(1, 6 * mm))
        story.append(Paragraph("Lowest Volume Products", sec_s))
        story.append(Paragraph("Consider whether these items justify their menu space, prep time, and ingredient costs.", note_s))
        bottom_products = forecast_df.groupby('Product_Name')['Forecast'].sum().sort_values().head(10).reset_index()
        bottom_table_data = [['Product', 'Monthly Total', 'Daily Avg']]
        for _, row in bottom_products.iterrows():
            bottom_table_data.append([str(row['Product_Name']), str(int(row['Forecast'])), f"{row['Forecast']/n_days:.1f}"])
        story.append(make_table(bottom_table_data, [5 * cm, 3 * cm, 3 * cm]))
        story.append(PageBreak())
        story.append(Paragraph("Day-of-Week Demand Patterns", sec_s))
        story.append(Paragraph("Average daily volume by day of week. Plan staffing rotas and prep schedules accordingly.", note_s))
        forecast_df['dow_name'] = forecast_df['Date'].dt.dayofweek.map({i: DAY_NAMES[i] for i in range(7)})
        day_averages = forecast_df.groupby('dow_name')['Forecast'].sum()
        day_counts = forecast_df.groupby('dow_name')['Date'].nunique()
        day_demand_data = [['Day', 'Total', 'Daily Avg']]
        for day_name in DAY_NAMES:
            if day_name in day_averages.index:
                day_demand_data.append([day_name, str(int(day_averages[day_name])), str(int(day_averages[day_name] / day_counts[day_name]))])
        story.append(make_table(day_demand_data, [4 * cm, 3 * cm, 3 * cm]))
        story.append(Spacer(1, 6 * mm))
        story.append(Paragraph("Monthly Ingredient Requirements", sec_s))
        story.append(Paragraph("Total ingredients for the month. Quantities over 500g shown in kg. Use for bulk ordering and supplier negotiations.", note_s))
        monthly_ingredients = (daily_ingredients.groupby(['ingredient', 'unit'])['total_qty'].sum().reset_index().sort_values('total_qty', ascending=False))
        monthly_ingredients = monthly_ingredients[monthly_ingredients['total_qty'] > 0]
        monthly_ingredients = convert_grams_to_kg(monthly_ingredients)
        ingredient_rows = [(str(row['ingredient']), fmt_qty(row['total_qty'], row['unit'])) for _, row in monthly_ingredients.iterrows()]
        if ingredient_rows:
            story.append(add_two_col(ingredient_rows, 'Ingredient', 'Monthly Total'))

    import time
    try:
        doc.build(story); print(f"    PDF saved: {output_path}")
    except PermissionError:
        ts = int(time.time())
        new_path = output_path.replace(".pdf", f"_{ts}.pdf")
        print(f"    WARNING: Permission denied on {output_path}. File might be open.")
        print(f"    Saving to fallback: {new_path}")
        from reportlab.lib.pagesizes import A4
        doc = SimpleDocTemplate(new_path, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        doc.build(story)
        print(f"    PDF saved: {new_path}")


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Multi-Model Forecast Pipeline')
    parser.add_argument('--model', help='Force a specific runner (xgb_improved_daily, arima_forecast, prophet_daily, etc.)')
    parser.add_argument('--november', action='store_true', help='Forecast November 2025 using notebook-style train/val/test split')
    args = parser.parse_args()

    print("\n" + "═"*58)
    print("  FORECAST PIPELINE v3 — All 9 Models")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("═"*58)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    recipes = get_recipes()

    # Priority: FORCE_MODEL variable > --model CLI arg > auto-select from DB
    forced = None
    if FORCE_MODEL is not None:
        forced = FORCE_MODEL.lower()
        print(f"\n  Forced model (from FORCE_MODEL variable): {forced}")
    elif args.model:
        forced = args.model.lower()
        print(f"\n  Forced model (from --model flag): {forced}")

    if forced:
        if forced not in MODEL_RUNNERS:
            print(f"  ERROR: '{forced}' is not a valid model name.")
            print(f"  Valid options: {', '.join(sorted(MODEL_RUNNERS.keys()))}")
            sys.exit(1)
        best_models = {h:{'model_type':forced,'wape':None,'run_id':'manual','mase':None,'mae':None,'bias':None} for h in ['day','week','month']}
        runners = {h:forced for h in ['day','week','month']}
    else:
        best_models = select_best_models(DB_PATH)
        if best_models is None:
            best_models = {h:None for h in ['day','week','month']}
            runners = {h:'xgb_improved_daily' for h in ['day','week','month']}
        else:
            runners = {}
            for h in ['day','week','month']:
                runners[h] = resolve_runner(best_models[h]['model_type']) if best_models[h] else 'xgb_improved_daily'

    if args.november:
        print("  MODE: November 2025 Evaluation (Notebook-style split)")
        # For November mode, notebook logic for train/val/test split
        # train until Oct 01, val until Nov 01, test until Nov 30
        # This results in a 29-day forecast starting Nov 02
        configs = [('month',29,'Monthly Stakeholder Report (November 2025)')]
    else:
        configs = [('day',1,'Daily Kitchen Prep Sheet'),('week',7,'Weekly Ordering Guide'),('month',30,'Monthly Stakeholder Report')]

    # Group by runner to avoid retraining
    runner_to_horizons = {}
    for horizon_key, horizon_days, report_title in configs:
        runner_name = runners[horizon_key]
        if runner_name not in runner_to_horizons: 
            runner_to_horizons[runner_name] = []
        runner_to_horizons[runner_name].append((horizon_key, horizon_days, report_title))

    forecasts = {}
    for runner_name, horizon_list in runner_to_horizons.items():
        max_days = max(hd for _, hd, _ in horizon_list)
        print(f"\n  ── Running {runner_name.upper()} (forecasting {max_days} days) ──")
        runner_func = MODEL_RUNNERS.get(runner_name)
        if not runner_func:
            print(f"    ERROR: No runner '{runner_name}'. Skipping.")
            continue
        
        if args.november:
            # Pass a flag or special handling for November
            # We'll modify runners to accept an optional 'eval_mode' or similar
            try:
                forecasts[runner_name] = runner_func(max_days, eval_mode='november')
            except TypeError:
                # Fallback if runner doesn't support eval_mode yet
                forecasts[runner_name] = runner_func(max_days)
        else:
            forecasts[runner_name] = runner_func(max_days)
        print(f"    Done: {len(forecasts[runner_name])} predictions")

    print(f"\n  ── Generating Reports ──")
    paths = []
    for horizon_key, horizon_days, report_title in configs:
        runner_name = runners[horizon_key]
        if runner_name not in forecasts: 
            continue
        full_forecast = forecasts[runner_name]
        dates = sorted(full_forecast['Date'].unique())[:horizon_days]
        forecast_df = full_forecast[full_forecast['Date'].isin(dates)].copy()
        merged = forecast_df.merge(recipes, left_on='Product_Name', right_on='product', how='left')
        merged['total_qty'] = merged['Forecast'] * merged['quantity']
        daily_ingredients = merged.groupby(['Date', 'ingredient', 'unit'])['total_qty'].sum().reset_index().sort_values(['Date', 'ingredient'])
        model_display_name = best_models[horizon_key]['model_type'] if best_models[horizon_key] else runner_name.upper()
        model_info = best_models[horizon_key]
        report_filename = f"forecast_{horizon_key}_{forecast_df['Date'].min().strftime('%Y-%m-%d')}.pdf"
        output_path = os.path.join(RESULTS_DIR, report_filename)
        print(f"    {report_title} ({model_display_name})...")
        generate_report(forecast_df, daily_ingredients, model_display_name, model_info, horizon_key, output_path)
        paths.append(output_path)

    if forecasts:
        longest_forecast = max(forecasts.values(), key=len)
        csv_path = os.path.join(RESULTS_DIR, f"forecast_all_{datetime.now().strftime('%Y%m%d')}.csv")
        longest_forecast.to_csv(csv_path, index=False)
        print(f"\n    CSV: {csv_path}")

    print("\n"+"═"*58)
    print("  PIPELINE COMPLETE — 3 reports generated")
    for p in paths: print(f"    → {p}")
    print("═"*58)


if __name__ == '__main__':
    main()
