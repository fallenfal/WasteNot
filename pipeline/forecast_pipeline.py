"""
Forecast pipeline - retrains a model, forecasts 1/7/30 days, and generates
3 PDF reports (kitchen prep, ordering, stakeholder).

By default it picks the best model from model_tracking.db based on WAPE.
Change FORCE_MODEL below to lock it to a specific model.

Usage:
    python forecast_pipeline.py
    python forecast_pipeline.py --model arima
"""

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
DATA_DIR    = '../preprocesing_data/processed_csv'
RESULTS_DIR = '../results'
DB_PATH     = os.path.join(RESULTS_DIR, 'model_tracking.db')

# how many optuna trials when retraining (30 is a good balance, drop to 10 for quick tests)
OPTUNA_TRIALS = 30

# store opens at 8, closes at 17, so 9 business hours
BUSINESS_HOURS = list(range(8, 17))
HOURS_PER_DAY  = 9

# ── which model to use ──
# Set to None => auto-picks best model from model_tracking.db (lowest WAPE per horizon)
# Or set to one of these to force it:
#   'xgb_simple_daily'       basic xgb, 3 lags
#   'xgb_improved_daily'     xgb with rolling/momentum features
#   'xgb_simple_hourly'      hourly xgb, 3 hourly lags, rolls up 9h to daily
#   'xgb_improved_hourly'    hourly xgb with rush flags + rolling
#   'arima'                  ARIMA(2,1,1), fastest option
#   'prophet_daily'          prophet per product
#   'prophet_hourly'         hourly prophet, rolls up to daily
#   'lstm_daily'             global LSTM, 30-day sequences (slow on CPU)
#   'lstm_hourly'            global hourly LSTM, 63-step sequences (slowest)
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


# ── recipes ──
# Try importing from the existing file first, otherwise build it inline.
# TODO: eventually move this to a CSV or DB table so the kitchen can update it
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
    """Check model_tracking.db for the winning model at each horizon."""
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
    """Figure out which runner function to call for a given DB model_type string."""
    r = MODEL_TYPE_TO_RUNNER.get(model_type)
    if r:
        return r
    # fuzzy match for model types I haven't explicitly mapped
    mt = model_type.lower()
    if 'xgboost' in mt and 'simple' in mt and 'hourly' in mt: return 'xgb_simple_hourly'
    if 'xgboost' in mt and 'hourly' in mt: return 'xgb_improved_hourly'
    if 'xgboost' in mt and 'simple' in mt: return 'xgb_simple_daily'
    if 'xgboost' in mt: return 'xgb_improved_daily'
    if 'lstm' in mt and 'hourly' in mt: return 'lstm_hourly'
    if 'lstm' in mt: return 'lstm_daily'
    if 'prophet' in mt and 'hourly' in mt: return 'prophet_hourly'
    if 'prophet' in mt: return 'prophet_daily'
    if 'arima' in mt: return 'arima'
    print(f"  WARNING: don't recognise model_type '{model_type}', falling back to xgb_improved_daily")
    return 'xgb_improved_daily'


# ── unit conversion ──
# anything over 500g gets shown as kg in the reports

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
    s = pd.read_csv(os.path.join(DATA_DIR, 'sales_data_preprocessed.csv'))
    s['Date'] = pd.to_datetime(s['Date']).dt.normalize()
    if 'Time' in s.columns:
        s = s.drop(columns=['Time'])
    ds = s.groupby('Date').sum(numeric_only=True).reset_index()
    product_cols = [c for c in ds.columns if c != 'Date']

    dl = pd.melt(ds, id_vars=['Date'], value_vars=product_cols,
                 var_name='Product_Name', value_name='Sales')
    dl['Sales'] = dl['Sales'].clip(lower=0)
    dl = dl[dl['Product_Name'].isin(PRODUCTS_TO_FORECAST)].reset_index(drop=True)
    dl = dl.sort_values(['Product_Name', 'Date']).reset_index(drop=True)
    return dl, ds

def load_sales_hourly():
    """Load hourly sales filtered to business hours only (8-16)."""
    s = pd.read_csv(os.path.join(DATA_DIR, 'sales_data_preprocessed.csv'))
    s['Date'] = pd.to_datetime(s['Date'])
    s = s[s['Date'].dt.hour.isin(BUSINESS_HOURS)]
    if 'Time' in s.columns:
        s = s.drop(columns=['Time'])
    pc = [c for c in s.columns if c not in ['Date', 'Time', 'date'] and s[c].dtype.kind in 'iufc']

    dl = pd.melt(s, id_vars=['Date'], value_vars=pc,
                 var_name='Product_Name', value_name='Sales')
    dl['Sales'] = dl['Sales'].clip(lower=0)
    dl = dl[dl['Product_Name'].isin(PRODUCTS_TO_FORECAST)].reset_index(drop=True)
    return dl.sort_values(['Product_Name', 'Date']).reset_index(drop=True)

def load_exogenous():
    # weather
    w = pd.read_csv(os.path.join(DATA_DIR, 'weather_data_hourly.csv'))
    w['Date'] = pd.to_datetime(w['Date']).dt.normalize()
    agg = {'apparent_temperature':'mean', 'precipitation':'sum', 'snowfall':'sum',
           'snow_depth':'max', 'relative_humidity_2m':'mean', 'cloud_cover':'mean',
           'visibility':'mean', 'wind_speed_10m':'mean', 'wind_gusts_10m':'max'}
    if 'weather_code' in w.columns:
        w['is_clear'] = (w['weather_code'] == 0).astype(int)
        w['is_cloudy'] = w['weather_code'].isin([1,2,3,45,48]).astype(int)
        w['is_rain'] = w['weather_code'].isin([51,53,55,56,57,61,63,65,66,67,80,81,82,95,96,99]).astype(int)
        w['is_snow'] = w['weather_code'].isin([71,73,75,77,85,86]).astype(int)
        agg.update({'is_clear':'max', 'is_cloudy':'max', 'is_rain':'max', 'is_snow':'max'})
    daily_weather = w.groupby('Date').agg(agg).reset_index()

    # holidays
    h = pd.read_csv(os.path.join(DATA_DIR, 'holidays_data_preprocessed.csv'))
    h['Date'] = pd.to_datetime(h['Date']).dt.normalize()
    daily_holidays = h.groupby('Date').max().reset_index()
    daily_holidays['is_holiday_lag_1'] = daily_holidays['is_holiday'].shift(1).fillna(0)
    daily_holidays['is_holiday_lead_1'] = daily_holidays['is_holiday'].shift(-1).fillna(0)

    # events
    e = pd.read_csv(os.path.join(DATA_DIR, 'aberdeen_events_master_timeline.csv'))
    e['Date'] = pd.to_datetime(e['Date']).dt.normalize()
    daily_events = e.groupby('Date').max(numeric_only=True).reset_index()

    return daily_weather, daily_holidays, daily_events


# ══════════════════════════════════════════
# MODEL RUNNERS
# each returns a DataFrame: Date, Product_Name, Forecast
# ══════════════════════════════════════════

def run_arima(forecast_days):
    from statsforecast import StatsForecast
    from statsforecast.models import ARIMA as AM
    print("    Loading data...")
    dl, _ = load_sales_long()
    ts = dl.rename(columns={'Product_Name':'unique_id','Date':'ds','Sales':'y'})
    print(f"    Training ARIMA(2,1,1) → {forecast_days} days...")
    sf = StatsForecast(models=[AM(order=(2,1,1),seasonal_order=(0,0,0),season_length=1,alias='ARIMA')],freq='D',n_jobs=-1)
    f = sf.forecast(df=ts[['unique_id','ds','y']], h=forecast_days)
    f['ARIMA'] = f['ARIMA'].clip(lower=0).round().astype(int)
    return f.rename(columns={'unique_id':'Product_Name','ds':'Date','ARIMA':'Forecast'})[['Date','Product_Name','Forecast']].reset_index(drop=True)

# ── 2. PROPHET DAILY ──
def run_prophet_daily(forecast_days):
    from prophet import Prophet
    import logging as lg
    lg.getLogger('prophet').setLevel(lg.WARNING)
    lg.getLogger('cmdstanpy').setLevel(lg.WARNING)
    print("    Loading data...")
    dl, _ = load_sales_long()
    dw, dh, de = load_exogenous()
    products = sorted(dl['Product_Name'].unique())
    results = []
    print(f"    Training Prophet for {len(products)} products → {forecast_days} days...")
    for p in products:
        pdf = dl[dl['Product_Name']==p][['Date','Sales']].rename(columns={'Date':'ds','Sales':'y'})
        pdf = pdf.merge(dh[['Date','is_holiday']].rename(columns={'Date':'ds'}), on='ds', how='left').fillna(0)
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                    seasonality_mode='additive', changepoint_prior_scale=0.05)
        m.add_country_holidays(country_name='GB')
        m.fit(pdf[['ds','y']])
        future = m.make_future_dataframe(periods=forecast_days)
        fc = m.predict(future)
        fc = fc.tail(forecast_days)[['ds','yhat']].copy()
        fc['yhat'] = fc['yhat'].clip(lower=0).round().astype(int)
        fc['Product_Name'] = p
        results.append(fc)
    out = pd.concat(results, ignore_index=True)
    return out.rename(columns={'ds':'Date','yhat':'Forecast'})[['Date','Product_Name','Forecast']]

# ── 3. PROPHET HOURLY ──
def run_prophet_hourly(forecast_days):
    from prophet import Prophet
    import logging as lg
    lg.getLogger('prophet').setLevel(lg.WARNING)
    lg.getLogger('cmdstanpy').setLevel(lg.WARNING)
    print("    Loading hourly data...")
    dl = load_sales_hourly()
    products = sorted(dl['Product_Name'].unique())
    results = []
    forecast_hours = forecast_days * HOURS_PER_DAY
    print(f"    Training Hourly Prophet for {len(products)} products → {forecast_days} days ({forecast_hours} hours)...")
    for p in products:
        pdf = dl[dl['Product_Name']==p][['Date','Sales']].rename(columns={'Date':'ds','Sales':'y'})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                    seasonality_mode='additive', changepoint_prior_scale=0.05)
        m.add_seasonality(name='daily_business', period=1, fourier_order=5)
        m.add_country_holidays(country_name='GB')
        m.fit(pdf[['ds','y']])
        future = m.make_future_dataframe(periods=forecast_hours, freq='h')
        future = future[future['ds'].dt.hour.isin(BUSINESS_HOURS)]
        fc = m.predict(future)
        fc = fc.tail(forecast_hours)[['ds','yhat']].copy()
        fc['yhat'] = fc['yhat'].clip(lower=0)
        fc['Product_Name'] = p
        results.append(fc)
    # Rollup hourly → daily
    out = pd.concat(results, ignore_index=True)
    out['Date'] = out['ds'].dt.normalize()
    daily = out.groupby(['Date','Product_Name'])['yhat'].sum().reset_index()
    daily['Forecast'] = daily['yhat'].round().astype(int)
    return daily[['Date','Product_Name','Forecast']]

# ── 4. XGB IMPROVED DAILY (same as v2) ──
def run_xgb_improved_daily(forecast_days):
    import xgboost as xgb
    import optuna; optuna.logging.set_verbosity(logging.WARNING)
    print("    Loading data...")
    dl, _ = load_sales_long(); dw, dh, de = load_exogenous()
    dl['Product_Name'] = dl['Product_Name'].astype('category')
    # Time features
    dl['day_of_week']=dl['Date'].dt.dayofweek
    dl['day_sin']=np.sin(2*np.pi*dl['day_of_week']/7); dl['day_cos']=np.cos(2*np.pi*dl['day_of_week']/7)
    dl['month']=dl['Date'].dt.month
    dl['month_sin']=np.sin(2*np.pi*(dl['month']-1)/12); dl['month_cos']=np.cos(2*np.pi*(dl['month']-1)/12)
    dl['day_of_month']=dl['Date'].dt.day; dl['Is_Weekend']=dl['Date'].dt.dayofweek.isin([5,6]).astype(int)
    dl['Year']=dl['Date'].dt.year; dl['week_of_year']=dl['Date'].dt.isocalendar().week.astype(int)
    df = dl.merge(dw,on='Date',how='left').merge(dh,on='Date',how='left').merge(de,on='Date',how='left')
    for c in ['date','Time']:
        if c in df.columns: df=df.drop(columns=[c])
    cc=df.select_dtypes(include=['category']).columns; df[df.columns.difference(cc)]=df[df.columns.difference(cc)].fillna(0)
    df=df.sort_values(['Product_Name','Date']).reset_index(drop=True)
    for lag in [1,2,7,14,30]: df[f'sales_{lag}_step_ago']=df.groupby('Product_Name',observed=False)['Sales'].shift(lag)
    for w in [3,7,14]:
        df[f'rolling_{w}d_avg']=df.groupby('Product_Name',observed=False)['sales_1_step_ago'].transform(lambda x:x.rolling(w,min_periods=1).mean())
        df[f'rolling_{w}d_std']=df.groupby('Product_Name',observed=False)['sales_1_step_ago'].transform(lambda x:x.rolling(w,min_periods=1).std()).fillna(0)
    df['sales_momentum']=df['sales_1_step_ago']-df['sales_7_step_ago']
    df['expanding_mean']=df.groupby('Product_Name',observed=False)['sales_1_step_ago'].transform(lambda x:x.expanding(min_periods=1).mean())
    df['ratio_1d_vs_7d']=df['sales_1_step_ago']/(df['rolling_7d_avg']+1e-8)
    df=df.dropna().reset_index(drop=True); df['Sales']=df['Sales'].clip(lower=0)
    fc=[c for c in df.columns if c not in ['Date','Sales']]
    md=df['Date'].max(); vs=md-pd.Timedelta(days=30)
    Xt,yt=df[df['Date']<=vs][fc],df[df['Date']<=vs]['Sales']
    Xv,yv=df[df['Date']>vs][fc],df[df['Date']>vs]['Sales']
    print(f"    Tuning XGBoost Improved ({OPTUNA_TRIALS} trials)...")
    def obj(trial):
        p={"n_estimators":1500,"early_stopping_rounds":50,"learning_rate":trial.suggest_float("lr",5e-3,0.1,log=True),
           "max_depth":trial.suggest_int("md",4,8),"min_child_weight":trial.suggest_int("mcw",2,8),
           "subsample":trial.suggest_float("ss",0.6,0.95),"colsample_bytree":trial.suggest_float("cb",0.5,0.9),
           "gamma":trial.suggest_float("g",1e-4,1.0,log=True),"reg_lambda":trial.suggest_float("rl",0.01,5.0,log=True),
           "reg_alpha":trial.suggest_float("ra",0.01,5.0,log=True),"enable_categorical":True,"tree_method":"hist"}
        m=xgb.XGBRegressor(**p); m.fit(Xt,yt,eval_set=[(Xv,yv)],verbose=False)
        return np.sqrt(np.mean((yv-m.predict(Xv))**2))
    study=optuna.create_study(direction="minimize"); study.optimize(obj,n_trials=OPTUNA_TRIALS)
    bp=study.best_params; bp.update({"n_estimators":1500,"early_stopping_rounds":30,"enable_categorical":True,"tree_method":"hist"})
    # Rename back to standard param names
    param_rename = {'lr':'learning_rate','md':'max_depth','mcw':'min_child_weight','ss':'subsample','cb':'colsample_bytree','g':'gamma','rl':'reg_lambda','ra':'reg_alpha'}
    bp = {param_rename.get(k,k):v for k,v in bp.items()}
    bp.pop('early_stopping_rounds', None)
    print(f"    Best RMSE: {study.best_value:.4f}. Training final model...")
    model=xgb.XGBRegressor(**bp); model.fit(df[fc],df['Sales'],verbose=False)
    # Recursive forecast
    return _xgb_recursive_forecast(df, model, fc, forecast_days,
        lag_map={1:'sales_1_step_ago',2:'sales_2_step_ago',7:'sales_7_step_ago',14:'sales_14_step_ago',30:'sales_30_step_ago'},
        rolling_windows=[3,7,14], use_momentum=True)

# ── 5. XGB SIMPLE DAILY ──
def run_xgb_simple_daily(forecast_days):
    import xgboost as xgb
    import optuna; optuna.logging.set_verbosity(logging.WARNING)
    print("    Loading data...")
    dl, _ = load_sales_long(); dw, dh, de = load_exogenous()
    dl['Product_Name'] = dl['Product_Name'].astype('category')
    df = dl.merge(dw,on='Date',how='left').merge(dh,on='Date',how='left').merge(de,on='Date',how='left')
    for c in ['date','Time']:
        if c in df.columns: df=df.drop(columns=[c])
    cc=df.select_dtypes(include=['category']).columns; df[df.columns.difference(cc)]=df[df.columns.difference(cc)].fillna(0)
    df=df.sort_values(['Product_Name','Date']).reset_index(drop=True)
    # Simple lags: 1, 7, 30 only
    for lag,name in [(1,'sales_1_step_ago'),(7,'sales_7_steps_ago'),(30,'sales_30_steps_ago')]:
        df[name]=df.groupby('Product_Name',observed=False)['Sales'].shift(lag)
    df=df.dropna().reset_index(drop=True); df['Sales']=df['Sales'].clip(lower=0)
    fc=[c for c in df.columns if c not in ['Date','Sales']]
    md=df['Date'].max(); vs=md-pd.Timedelta(days=30)
    Xt,yt=df[df['Date']<=vs][fc],df[df['Date']<=vs]['Sales']
    Xv,yv=df[df['Date']>vs][fc],df[df['Date']>vs]['Sales']
    print(f"    Tuning XGBoost Simple ({OPTUNA_TRIALS} trials)...")
    def obj(trial):
        p={"n_estimators":1000,"early_stopping_rounds":50,"learning_rate":trial.suggest_float("lr",5e-3,0.1,log=True),
           "max_depth":trial.suggest_int("md",3,7),"min_child_weight":trial.suggest_int("mcw",2,8),
           "subsample":trial.suggest_float("ss",0.6,0.95),"colsample_bytree":trial.suggest_float("cb",0.5,0.9),
           "gamma":trial.suggest_float("g",1e-4,1.0,log=True),"reg_lambda":trial.suggest_float("rl",0.01,5.0,log=True),
           "reg_alpha":trial.suggest_float("ra",0.01,5.0,log=True),"enable_categorical":True,"tree_method":"hist"}
        m=xgb.XGBRegressor(**p); m.fit(Xt,yt,eval_set=[(Xv,yv)],verbose=False)
        return np.sqrt(np.mean((yv-m.predict(Xv))**2))
    study=optuna.create_study(direction="minimize"); study.optimize(obj,n_trials=OPTUNA_TRIALS)
    bp=study.best_params; bp.update({"n_estimators":1000,"early_stopping_rounds":30,"enable_categorical":True,"tree_method":"hist"})
    param_rename = {'lr':'learning_rate','md':'max_depth','mcw':'min_child_weight','ss':'subsample','cb':'colsample_bytree','g':'gamma','rl':'reg_lambda','ra':'reg_alpha'}
    bp = {param_rename.get(k,k):v for k,v in bp.items()}
    bp.pop('early_stopping_rounds', None)
    print(f"    Best RMSE: {study.best_value:.4f}. Training final model...")
    model=xgb.XGBRegressor(**bp); model.fit(df[fc],df['Sales'],verbose=False)
    return _xgb_recursive_forecast(df, model, fc, forecast_days,
        lag_map={1:'sales_1_step_ago',7:'sales_7_steps_ago',30:'sales_30_steps_ago'},
        rolling_windows=[], use_momentum=False)

# ── SHARED XGB RECURSIVE FORECAST HELPER ──
def _xgb_recursive_forecast(df, model, feature_cols, forecast_days, lag_map, rolling_windows, use_momentum):
    """Shared recursive daily XGBoost forecast logic."""
    last_date = df['Date'].max()
    forecast_dates = [last_date + pd.Timedelta(days=i+1) for i in range(forecast_days)]
    products = sorted(PRODUCTS_TO_FORECAST)
    sales_history = {(r['Product_Name'],r['Date']):r['Sales'] for _,r in df[['Date','Product_Name','Sales']].iterrows()}
    rows = [{'Date':d,'Product_Name':p} for d in forecast_dates for p in products]
    fdf = pd.DataFrame(rows)
    fdf['Product_Name'] = fdf['Product_Name'].astype(pd.CategoricalDtype(categories=df['Product_Name'].cat.categories))
    # Copy time features
    for tc in ['day_of_week','day_sin','day_cos','month','month_sin','month_cos','day_of_month','Is_Weekend','Year','week_of_year']:
        if tc in feature_cols:
            if tc=='day_of_week': fdf[tc]=fdf['Date'].dt.dayofweek
            elif tc=='day_sin': fdf[tc]=np.sin(2*np.pi*fdf['Date'].dt.dayofweek/7)
            elif tc=='day_cos': fdf[tc]=np.cos(2*np.pi*fdf['Date'].dt.dayofweek/7)
            elif tc=='month': fdf[tc]=fdf['Date'].dt.month
            elif tc=='month_sin': fdf[tc]=np.sin(2*np.pi*(fdf['Date'].dt.month-1)/12)
            elif tc=='month_cos': fdf[tc]=np.cos(2*np.pi*(fdf['Date'].dt.month-1)/12)
            elif tc=='day_of_month': fdf[tc]=fdf['Date'].dt.day
            elif tc=='Is_Weekend': fdf[tc]=fdf['Date'].dt.dayofweek.isin([5,6]).astype(int)
            elif tc=='Year': fdf[tc]=fdf['Date'].dt.year
            elif tc=='week_of_year': fdf[tc]=fdf['Date'].dt.isocalendar().week.astype(int)
    # Exogenous: last known
    exog_exclude = set(['Date','Sales','Product_Name','day_of_week','day_sin','day_cos','month',
        'month_sin','month_cos','day_of_month','Is_Weekend','Year','week_of_year'])
    exog_exclude.update(lag_map.values())
    for w in rolling_windows: exog_exclude.update([f'rolling_{w}d_avg',f'rolling_{w}d_std'])
    if use_momentum: exog_exclude.update(['sales_momentum','expanding_mean','ratio_1d_vs_7d'])
    exog_cols = [c for c in df.columns if c not in exog_exclude and c in feature_cols]
    if exog_cols:
        last_exog = df[df['Date']==last_date][exog_cols].iloc[0]
        for c in exog_cols: fdf[c] = last_exog[c]
    # Init lags
    for idx,row in fdf.iterrows():
        p,d = row['Product_Name'],row['Date']
        for ld,lc in lag_map.items(): fdf.at[idx,lc]=sales_history.get((p,d-pd.Timedelta(days=ld)),0)
        if rolling_windows or use_momentum:
            recent=[sales_history.get((p,d-pd.Timedelta(days=i)),0) for i in range(1,15)]
            for w in rolling_windows:
                win=recent[:w]
                fdf.at[idx,f'rolling_{w}d_avg']=np.mean(win)
                fdf.at[idx,f'rolling_{w}d_std']=np.std(win,ddof=1) if len(win)>1 else 0
            if use_momentum:
                fdf.at[idx,'sales_momentum']=fdf.at[idx,list(lag_map.values())[0]]-fdf.at[idx,lag_map.get(7,list(lag_map.values())[-1])]
                fdf.at[idx,'expanding_mean']=np.mean(recent)
                r7=fdf.at[idx,'rolling_7d_avg'] if 'rolling_7d_avg' in fdf.columns else 0
                fdf.at[idx,'ratio_1d_vs_7d']=fdf.at[idx,list(lag_map.values())[0]]/(r7+1e-8)
    for c in feature_cols:
        if c not in fdf.columns: fdf[c]=0
    for c in fdf.select_dtypes(include=[np.number]).columns:
        fdf[c] = fdf[c].fillna(0)
    idx_lookup={(r['Product_Name'],r['Date']):i for i,r in fdf.iterrows()}
    for di,cd in enumerate(forecast_dates):
        didx=fdf.index[fdf['Date']==cd].tolist()
        preds=np.clip(model.predict(fdf.loc[didx,feature_cols]),0,None).round().astype(int)
        fdf.loc[didx,'Forecast']=preds
        for ri,pv in zip(didx,preds):
            product=fdf.at[ri,'Product_Name']; sales_history[(product,cd)]=pv
            for ld,lc in lag_map.items():
                fk=(product,cd+pd.Timedelta(days=ld))
                if fk in idx_lookup: fdf.at[idx_lookup[fk],lc]=pv
        if di+1<len(forecast_dates):
            nd=forecast_dates[di+1]
            for ni in fdf.index[fdf['Date']==nd]:
                p=fdf.at[ni,'Product_Name']
                recent=[sales_history.get((p,nd-pd.Timedelta(days=i)),0) for i in range(1,15)]
                for w in rolling_windows:
                    win=recent[:w]
                    fdf.at[ni,f'rolling_{w}d_avg']=np.mean(win)
                    fdf.at[ni,f'rolling_{w}d_std']=np.std(win,ddof=1) if len(win)>1 else 0
                if use_momentum:
                    fdf.at[ni,'sales_momentum']=fdf.at[ni,list(lag_map.values())[0]]-fdf.at[ni,lag_map.get(7,list(lag_map.values())[-1])]
                    fdf.at[ni,'expanding_mean']=np.mean(recent)
                    fdf.at[ni,'ratio_1d_vs_7d']=fdf.at[ni,list(lag_map.values())[0]]/(fdf.at[ni,'rolling_7d_avg']+1e-8)
    result=fdf[['Date','Product_Name','Forecast']].copy(); result['Forecast']=result['Forecast'].astype(int)
    return result

# ── 6 & 7. XGB HOURLY (simple + improved) ──
def run_xgb_simple_hourly(forecast_days):
    """Simple XGBoost hourly: 3 hourly lags, predict per-hour then aggregate to daily."""
    import xgboost as xgb
    import optuna; optuna.logging.set_verbosity(logging.WARNING)
    print("    Loading hourly data...")
    dl = load_sales_hourly()
    dl['Product_Name'] = dl['Product_Name'].astype('category')
    dl = dl.sort_values(['Product_Name','Date']).reset_index(drop=True)
    # Simple hourly lags
    for lag,name in [(1,'sales_1h_ago'),(HOURS_PER_DAY,'sales_same_hour_yesterday'),(HOURS_PER_DAY*7,'sales_same_hour_last_week')]:
        dl[name]=dl.groupby('Product_Name',observed=False)['Sales'].shift(lag)
    # Time features
    dl['hour_of_day']=dl['Date'].dt.hour; dl['day_of_week']=dl['Date'].dt.dayofweek
    dl['Is_Weekend']=dl['day_of_week'].isin([5,6]).astype(int)
    dl=dl.dropna().reset_index(drop=True); dl['Sales']=dl['Sales'].clip(lower=0)
    fc=[c for c in dl.columns if c not in ['Date','Sales']]
    md=dl['Date'].max(); vs=md-pd.Timedelta(days=30)
    Xt,yt=dl[dl['Date']<=vs][fc],dl[dl['Date']<=vs]['Sales']
    Xv,yv=dl[dl['Date']>vs][fc],dl[dl['Date']>vs]['Sales']
    print(f"    Tuning Simple Hourly XGBoost ({OPTUNA_TRIALS} trials)...")
    def obj(trial):
        p={"n_estimators":1000,"early_stopping_rounds":50,"learning_rate":trial.suggest_float("lr",5e-3,0.1,log=True),
           "max_depth":trial.suggest_int("md",3,7),"min_child_weight":trial.suggest_int("mcw",2,8),
           "subsample":trial.suggest_float("ss",0.6,0.95),"colsample_bytree":trial.suggest_float("cb",0.5,0.9),
           "gamma":trial.suggest_float("g",1e-4,1.0,log=True),"reg_lambda":trial.suggest_float("rl",0.01,5.0,log=True),
           "reg_alpha":trial.suggest_float("ra",0.01,5.0,log=True),"enable_categorical":True,"tree_method":"hist"}
        m=xgb.XGBRegressor(**p); m.fit(Xt,yt,eval_set=[(Xv,yv)],verbose=False)
        return np.sqrt(np.mean((yv-m.predict(Xv))**2))
    study=optuna.create_study(direction="minimize"); study.optimize(obj,n_trials=OPTUNA_TRIALS)
    bp=study.best_params; bp.update({"n_estimators":1000,"early_stopping_rounds":30,"enable_categorical":True,"tree_method":"hist"})
    param_rename = {'lr':'learning_rate','md':'max_depth','mcw':'min_child_weight','ss':'subsample','cb':'colsample_bytree','g':'gamma','rl':'reg_lambda','ra':'reg_alpha'}
    bp = {param_rename.get(k,k):v for k,v in bp.items()}
    bp.pop('early_stopping_rounds', None)
    print(f"    Best RMSE: {study.best_value:.4f}. Training + forecasting...")
    model=xgb.XGBRegressor(**bp); model.fit(dl[fc],dl['Sales'],verbose=False)
    # Hourly recursive forecast → aggregate to daily
    return _xgb_hourly_recursive(dl, model, fc, forecast_days,
        lag_map={1:'sales_1h_ago',HOURS_PER_DAY:'sales_same_hour_yesterday',HOURS_PER_DAY*7:'sales_same_hour_last_week'},
        rolling_windows=[])

def run_xgb_improved_hourly(forecast_days):
    """Improved XGBoost hourly: 5 hourly lags, rolling, rush flags."""
    import xgboost as xgb
    import optuna; optuna.logging.set_verbosity(logging.WARNING)
    print("    Loading hourly data...")
    dl = load_sales_hourly()
    dl['Product_Name'] = dl['Product_Name'].astype('category')
    dl = dl.sort_values(['Product_Name','Date']).reset_index(drop=True)
    # Improved hourly lags
    for lag,name in [(1,'sales_1h_ago'),(2,'sales_2h_ago'),(HOURS_PER_DAY,'sales_same_hour_yesterday'),
                     (HOURS_PER_DAY*7,'sales_same_hour_last_week'),(HOURS_PER_DAY*14,'sales_same_hour_2weeks_ago')]:
        dl[name]=dl.groupby('Product_Name',observed=False)['Sales'].shift(lag)
    # Rolling
    for w,wn in [(HOURS_PER_DAY,'1d'),(HOURS_PER_DAY*3,'3d'),(HOURS_PER_DAY*7,'7d')]:
        dl[f'rolling_{wn}_avg']=dl.groupby('Product_Name',observed=False)['sales_1h_ago'].transform(lambda x:x.rolling(w,min_periods=1).mean())
        dl[f'rolling_{wn}_std']=dl.groupby('Product_Name',observed=False)['sales_1h_ago'].transform(lambda x:x.rolling(w,min_periods=1).std()).fillna(0)
    # Time + rush features
    dl['hour_of_day']=dl['Date'].dt.hour
    dl['hour_sin']=np.sin(2*np.pi*(dl['hour_of_day']-8)/9)
    dl['hour_cos']=np.cos(2*np.pi*(dl['hour_of_day']-8)/9)
    dl['day_of_week']=dl['Date'].dt.dayofweek
    dl['day_sin']=np.sin(2*np.pi*dl['day_of_week']/7)
    dl['day_cos']=np.cos(2*np.pi*dl['day_of_week']/7)
    dl['month']=dl['Date'].dt.month
    dl['month_sin']=np.sin(2*np.pi*(dl['month']-1)/12)
    dl['month_cos']=np.cos(2*np.pi*(dl['month']-1)/12)
    dl['Is_Weekend']=dl['day_of_week'].isin([5,6]).astype(int)
    dl['is_morning_rush']=dl['hour_of_day'].isin([8,9,10]).astype(int)
    dl['is_lunch_rush']=dl['hour_of_day'].isin([11,12,13,14]).astype(int)
    dl['is_afternoon']=dl['hour_of_day'].isin([15,16]).astype(int)
    dl=dl.dropna().reset_index(drop=True); dl['Sales']=dl['Sales'].clip(lower=0)
    fc=[c for c in dl.columns if c not in ['Date','Sales']]
    md=dl['Date'].max(); vs=md-pd.Timedelta(days=30)
    Xt,yt=dl[dl['Date']<=vs][fc],dl[dl['Date']<=vs]['Sales']
    Xv,yv=dl[dl['Date']>vs][fc],dl[dl['Date']>vs]['Sales']
    print(f"    Tuning Improved Hourly XGBoost ({OPTUNA_TRIALS} trials)...")
    def obj(trial):
        p={"n_estimators":2000,"early_stopping_rounds":50,"learning_rate":trial.suggest_float("lr",3e-3,0.15,log=True),
           "max_depth":trial.suggest_int("md",4,10),"min_child_weight":trial.suggest_int("mcw",2,10),
           "subsample":trial.suggest_float("ss",0.6,0.95),"colsample_bytree":trial.suggest_float("cb",0.4,0.9),
           "gamma":trial.suggest_float("g",1e-4,2.0,log=True),"reg_lambda":trial.suggest_float("rl",0.01,10.0,log=True),
           "reg_alpha":trial.suggest_float("ra",0.01,10.0,log=True),"enable_categorical":True,"tree_method":"hist"}
        m=xgb.XGBRegressor(**p); m.fit(Xt,yt,eval_set=[(Xv,yv)],verbose=False)
        return np.sqrt(np.mean((yv-m.predict(Xv))**2))
    study=optuna.create_study(direction="minimize"); study.optimize(obj,n_trials=OPTUNA_TRIALS)
    bp=study.best_params; bp.update({"n_estimators":2000,"early_stopping_rounds":30,"enable_categorical":True,"tree_method":"hist"})
    param_rename = {'lr':'learning_rate','md':'max_depth','mcw':'min_child_weight','ss':'subsample','cb':'colsample_bytree','g':'gamma','rl':'reg_lambda','ra':'reg_alpha'}
    bp = {param_rename.get(k,k):v for k,v in bp.items()}
    bp.pop('early_stopping_rounds', None)
    print(f"    Best RMSE: {study.best_value:.4f}. Training + forecasting...")
    model=xgb.XGBRegressor(**bp); model.fit(dl[fc],dl['Sales'],verbose=False)
    return _xgb_hourly_recursive(dl, model, fc, forecast_days,
        lag_map={1:'sales_1h_ago',2:'sales_2h_ago',HOURS_PER_DAY:'sales_same_hour_yesterday',
                 HOURS_PER_DAY*7:'sales_same_hour_last_week',HOURS_PER_DAY*14:'sales_same_hour_2weeks_ago'},
        rolling_windows=[(HOURS_PER_DAY,'1d'),(HOURS_PER_DAY*3,'3d'),(HOURS_PER_DAY*7,'7d')])

def _xgb_hourly_recursive(dl, model, feature_cols, forecast_days, lag_map, rolling_windows):
    """Shared hourly XGBoost recursive forecast → aggregate to daily."""
    last_dt = dl['Date'].max()
    last_date = last_dt.normalize()
    # Build future hourly timestamps
    forecast_hours = []
    for d in range(forecast_days):
        day = last_date + pd.Timedelta(days=d+1)
        for h in BUSINESS_HOURS:
            forecast_hours.append(day + pd.Timedelta(hours=h))

    products = sorted(dl['Product_Name'].cat.categories)
    # Build sales history
    sales_hist = {(r['Product_Name'],r['Date']):r['Sales'] for _,r in dl[['Date','Product_Name','Sales']].iterrows()}

    rows = [{'Date':dt,'Product_Name':p} for dt in forecast_hours for p in products]
    fdf = pd.DataFrame(rows)
    fdf['Product_Name'] = fdf['Product_Name'].astype(pd.CategoricalDtype(categories=dl['Product_Name'].cat.categories))

    # Time features
    for c in feature_cols:
        if c not in fdf.columns:
            if c=='hour_of_day': fdf[c]=fdf['Date'].dt.hour
            elif c=='hour_sin': fdf[c]=np.sin(2*np.pi*(fdf['Date'].dt.hour-8)/9)
            elif c=='hour_cos': fdf[c]=np.cos(2*np.pi*(fdf['Date'].dt.hour-8)/9)
            elif c=='day_of_week': fdf[c]=fdf['Date'].dt.dayofweek
            elif c=='day_sin': fdf[c]=np.sin(2*np.pi*fdf['Date'].dt.dayofweek/7)
            elif c=='day_cos': fdf[c]=np.cos(2*np.pi*fdf['Date'].dt.dayofweek/7)
            elif c=='month': fdf[c]=fdf['Date'].dt.month
            elif c=='month_sin': fdf[c]=np.sin(2*np.pi*(fdf['Date'].dt.month-1)/12)
            elif c=='month_cos': fdf[c]=np.cos(2*np.pi*(fdf['Date'].dt.month-1)/12)
            elif c=='Is_Weekend': fdf[c]=fdf['Date'].dt.dayofweek.isin([5,6]).astype(int)
            elif c=='is_morning_rush': fdf[c]=fdf['Date'].dt.hour.isin([8,9,10]).astype(int)
            elif c=='is_lunch_rush': fdf[c]=fdf['Date'].dt.hour.isin([11,12,13,14]).astype(int)
            elif c=='is_afternoon': fdf[c]=fdf['Date'].dt.hour.isin([15,16]).astype(int)
            else: fdf[c]=0

    # Init lags from history
    for idx,row in fdf.iterrows():
        p,dt = row['Product_Name'],row['Date']
        for ls,lc in lag_map.items():
            past = dt - pd.Timedelta(hours=ls)
            fdf.at[idx,lc] = sales_hist.get((p,past),0)
        for ws,wn in rolling_windows:
            vals=[sales_hist.get((p,dt-pd.Timedelta(hours=i)),0) for i in range(1,ws+1)]
            fdf.at[idx,f'rolling_{wn}_avg']=np.mean(vals) if vals else 0
            fdf.at[idx,f'rolling_{wn}_std']=np.std(vals,ddof=1) if len(vals)>1 else 0
    for c in fdf.select_dtypes(include=[np.number]).columns:
        fdf[c] = fdf[c].fillna(0)

    idx_lookup={(r['Product_Name'],r['Date']):i for i,r in fdf.iterrows()}

    # Predict hour by hour
    for hi,cur_dt in enumerate(forecast_hours):
        hidx=fdf.index[fdf['Date']==cur_dt].tolist()
        preds=np.clip(model.predict(fdf.loc[hidx,feature_cols]),0,None)
        fdf.loc[hidx,'Forecast']=preds
        for ri,pv in zip(hidx,preds):
            product=fdf.at[ri,'Product_Name']; sales_hist[(product,cur_dt)]=pv
            for ls,lc in lag_map.items():
                fk=(product,cur_dt+pd.Timedelta(hours=ls))
                if fk in idx_lookup: fdf.at[idx_lookup[fk],lc]=pv
        # Update rolling for next hour
        if hi+1<len(forecast_hours):
            ndt=forecast_hours[hi+1]
            for ni in fdf.index[fdf['Date']==ndt]:
                p=fdf.at[ni,'Product_Name']
                for ws,wn in rolling_windows:
                    vals=[sales_hist.get((p,ndt-pd.Timedelta(hours=i)),0) for i in range(1,ws+1)]
                    fdf.at[ni,f'rolling_{wn}_avg']=np.mean(vals) if vals else 0
                    fdf.at[ni,f'rolling_{wn}_std']=np.std(vals,ddof=1) if len(vals)>1 else 0

    # Aggregate to daily
    fdf['Date_Day']=fdf['Date'].dt.normalize()
    daily=fdf.groupby(['Date_Day','Product_Name'])['Forecast'].sum().reset_index()
    daily=daily.rename(columns={'Date_Day':'Date'})
    daily['Forecast']=daily['Forecast'].round().astype(int)
    return daily[['Date','Product_Name','Forecast']]

# ── 8. LSTM GLOBAL DAILY (CPU-compatible) ──
def run_lstm_daily(forecast_days):
    """Global LSTM daily: one model for all products, 30-day sequences, recursive forecast."""
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

    SEQ_LEN = 30
    OPTUNA_LSTM_TRIALS = 10  # Fewer trials for CPU speed

    print("    Loading data for LSTM Daily (CPU mode)...")
    dl, daily_sales = load_sales_long()
    dw, dh, de = load_exogenous()

    # Build wide format with time + exogenous features
    product_cols = [c for c in daily_sales.columns if c != 'Date']
    daily_sales['day_sin'] = np.sin(2*np.pi*daily_sales['Date'].dt.dayofweek/7)
    daily_sales['day_cos'] = np.cos(2*np.pi*daily_sales['Date'].dt.dayofweek/7)
    daily_sales['month_sin'] = np.sin(2*np.pi*(daily_sales['Date'].dt.month-1)/12)
    daily_sales['month_cos'] = np.cos(2*np.pi*(daily_sales['Date'].dt.month-1)/12)
    daily_sales['Is_Weekend'] = daily_sales['Date'].dt.dayofweek.isin([5,6]).astype(int)
    time_feats = ['day_sin','day_cos','month_sin','month_cos','Is_Weekend']

    exclude = ['Date','Time','date']
    w_feats = [c for c in dw.columns if c not in exclude]
    h_feats = [c for c in dh.columns if c not in exclude]
    e_feats = [c for c in de.columns if c not in exclude]

    df_wide = daily_sales.merge(dw, on='Date', how='left')
    df_wide = df_wide.merge(dh, on='Date', how='left')
    df_wide = df_wide.merge(de, on='Date', how='left')
    for c in df_wide.select_dtypes(include=[np.number]).columns:
        df_wide[c] = df_wide[c].fillna(0)

    base_feat_cols = [c for c in w_feats + h_feats + e_feats + time_feats
                      if c in df_wide.columns and df_wide[c].dtype.kind in 'iufc']

    # Melt to long
    dfl = pd.melt(df_wide, id_vars=['Date']+base_feat_cols, value_vars=product_cols,
                  var_name='Product_Name', value_name='Sales')
    dfl['Sales'] = dfl['Sales'].clip(lower=0)
    dfl = dfl[dfl['Product_Name'].isin(PRODUCTS_TO_FORECAST)]
    dfl = dfl.sort_values(['Product_Name','Date']).reset_index(drop=True)

    enc = LabelEncoder()
    dfl['product_id'] = enc.fit_transform(dfl['Product_Name'])

    # Lag features
    dfl['sales_lag_1'] = dfl.groupby('Product_Name')['Sales'].shift(1)
    dfl['sales_lag_7'] = dfl.groupby('Product_Name')['Sales'].shift(7)
    dfl['sales_rolling_7_mean'] = dfl.groupby('Product_Name')['Sales'].shift(1).groupby(
        dfl['Product_Name']).transform(lambda x: x.rolling(7, min_periods=1).mean())
    dfl['sales_rolling_7_std'] = dfl.groupby('Product_Name')['Sales'].shift(1).groupby(
        dfl['Product_Name']).transform(lambda x: x.rolling(7, min_periods=1).std()).fillna(0)
    dfl['sales_diff_1'] = dfl.groupby('Product_Name')['Sales'].diff(1)

    lag_feats = ['sales_lag_1','sales_lag_7','sales_rolling_7_mean','sales_rolling_7_std','sales_diff_1']
    dfl = dfl.dropna(subset=lag_feats).reset_index(drop=True)
    feature_cols = base_feat_cols + ['product_id'] + lag_feats

    # Split
    max_date = dfl['Date'].max()
    val_start = max_date - pd.Timedelta(days=60)
    val_end_dt = max_date - pd.Timedelta(days=30)

    feat_scaler = MinMaxScaler(); tgt_scaler = MinMaxScaler()
    train_data = dfl[dfl['Date'] <= val_start]
    feat_scaler.fit(train_data[feature_cols])
    tgt_scaler.fit(train_data[['Sales']])

    # Build sequences
    def build_seqs(pdf):
        if len(pdf) < SEQ_LEN+1: return None, None, None
        sf = feat_scaler.transform(pdf[feature_cols])
        st = tgt_scaler.transform(pdf[['Sales']]).flatten()
        X, y, d = [], [], []
        for i in range(len(sf)-SEQ_LEN):
            X.append(sf[i:i+SEQ_LEN]); y.append(st[i+SEQ_LEN]); d.append(pdf['Date'].values[i+SEQ_LEN])
        return np.array(X), np.array(y), pd.to_datetime(d)

    all_Xt, all_yt, all_Xv, all_yv = [], [], [], []
    for pn in dfl['Product_Name'].unique():
        pdf = dfl[dfl['Product_Name']==pn].sort_values('Date')
        res = build_seqs(pdf)
        if res[0] is None: continue
        X, y, dates = res
        tm = dates <= np.datetime64(val_start)
        vm = (dates > np.datetime64(val_start)) & (dates <= np.datetime64(val_end_dt))
        if tm.sum()>0: all_Xt.append(X[tm]); all_yt.append(y[tm])
        if vm.sum()>0: all_Xv.append(X[vm]); all_yv.append(y[vm])

    X_train = np.vstack(all_Xt); y_train = np.concatenate(all_yt)
    X_val = np.vstack(all_Xv); y_val = np.concatenate(all_yv)
    shuf = np.random.permutation(len(X_train))
    X_train, y_train = X_train[shuf], y_train[shuf]

    print(f"    Sequences: {X_train.shape[0]} train, {X_val.shape[0]} val")
    print(f"    Tuning LSTM ({OPTUNA_LSTM_TRIALS} trials, CPU)...")

    def objective(trial):
        tf.keras.backend.clear_session()
        bs = trial.suggest_categorical('batch_size',[64,128,256])
        u1 = trial.suggest_int('lstm_1_units',64,192,step=32)
        u2 = trial.suggest_int('lstm_2_units',32,96,step=16)
        dr = trial.suggest_float('dropout_rate',0.1,0.4)
        lr = trial.suggest_float('learning_rate',1e-4,1e-2,log=True)
        l2r = trial.suggest_float('l2_reg',1e-5,1e-2,log=True)
        m = Sequential([Input(shape=(SEQ_LEN,len(feature_cols))),
            LSTM(u1,activation='tanh',return_sequences=True), Dropout(dr),
            LSTM(u2,activation='tanh'), Dropout(dr),
            Dense(64,activation='relu',kernel_regularizer=l2(l2r)),
            Dense(32,activation='relu'), Dense(1)])
        m.compile(optimizer=Adam(learning_rate=lr), loss=Huber())
        m.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=50, batch_size=bs,
              callbacks=[EarlyStopping(patience=8,restore_best_weights=True)], verbose=0)
        vp = tgt_scaler.inverse_transform(m.predict(X_val,verbose=0)).flatten()
        va = tgt_scaler.inverse_transform(y_val.reshape(-1,1)).flatten()
        return np.sqrt(np.mean((va-vp)**2))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=OPTUNA_LSTM_TRIALS)
    bp = study.best_params
    print(f"    Best RMSE: {study.best_value:.4f}. Training final model...")

    tf.keras.backend.clear_session()
    final = Sequential([Input(shape=(SEQ_LEN,len(feature_cols))),
        LSTM(bp['lstm_1_units'],activation='tanh',return_sequences=True), Dropout(bp['dropout_rate']),
        LSTM(bp['lstm_2_units'],activation='tanh'), Dropout(bp['dropout_rate']),
        Dense(64,activation='relu',kernel_regularizer=l2(bp['l2_reg'])),
        Dense(32,activation='relu'), Dense(1)])
    final.compile(optimizer=Adam(learning_rate=bp['learning_rate']), loss=Huber())
    final.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=150,
              batch_size=bp['batch_size'],
              callbacks=[EarlyStopping(patience=15,restore_best_weights=True),
                         ReduceLROnPlateau(factor=0.5,patience=5,min_lr=1e-6)], verbose=0)

    # Recursive forecast
    print(f"    Forecasting {forecast_days} days recursively...")
    lag_idx = [feature_cols.index(f) for f in lag_feats]
    lag_min = feat_scaler.data_min_[lag_idx]
    lag_range = feat_scaler.data_range_[lag_idx]
    lag_range[lag_range==0] = 1.0

    results = []
    for pn in sorted(dfl['Product_Name'].unique()):
        pdf = dfl[dfl['Product_Name']==pn].sort_values('Date')
        if len(pdf) < SEQ_LEN: continue
        tail = pdf.tail(SEQ_LEN)
        seq = feat_scaler.transform(tail[feature_cols])
        sales_hist = list(pdf['Sales'].values[-SEQ_LEN:])

        for day_i in range(forecast_days):
            pred_s = final.predict(seq.reshape(1,SEQ_LEN,-1), verbose=0)[0,0]
            pred_r = max(0, tgt_scaler.inverse_transform([[pred_s]])[0,0])
            sales_hist.append(pred_r)
            fd = max_date + pd.Timedelta(days=day_i+1)
            results.append({'Date':fd,'Product_Name':pn,'Forecast':int(round(pred_r))})

            if day_i+1 < forecast_days:
                next_feat = seq[-1].copy()
                # Update time features
                nd = fd + pd.Timedelta(days=1); dow=nd.dayofweek; mon=nd.month
                tv = [np.sin(2*np.pi*dow/7),np.cos(2*np.pi*dow/7),
                      np.sin(2*np.pi*(mon-1)/12),np.cos(2*np.pi*(mon-1)/12),
                      1 if dow>=5 else 0]
                for ti,tf_name in enumerate(time_feats):
                    if tf_name in feature_cols:
                        fi = feature_cols.index(tf_name)
                        next_feat[fi] = (tv[ti]-feat_scaler.data_min_[fi])/(feat_scaler.data_range_[fi]+1e-8)
                # Update lags from sales_hist
                h = sales_hist
                raw = np.array([h[-1], h[-7] if len(h)>=7 else h[0],
                    np.mean(h[-7:]) if len(h)>=7 else np.mean(h),
                    np.std(h[-7:]) if len(h)>=7 else 0.0,
                    h[-1]-h[-2] if len(h)>=2 else 0.0])
                scaled = (raw-lag_min)/lag_range
                for li,fi in enumerate(lag_idx): next_feat[fi]=scaled[li]
                seq = np.roll(seq,-1,axis=0); seq[-1]=next_feat

    tf.keras.backend.clear_session()
    return pd.DataFrame(results)


# ── 9. LSTM GLOBAL HOURLY (CPU-compatible) ──
def run_lstm_hourly(forecast_days):
    """Global LSTM hourly: one model for all products, 63-step sequences, 9h→daily rollup."""
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

    SEQ_LEN = 63  # ~1 week of hourly data (9h × 7d)
    OPTUNA_LSTM_TRIALS = 8

    print("    Loading hourly data for LSTM Hourly (CPU mode)...")
    dlh = load_sales_hourly()

    # Load exogenous (hourly weather already in sales file; for simplicity use daily agg)
    dw, dh, de = load_exogenous()

    # Time features on hourly data
    dlh['day_sin'] = np.sin(2*np.pi*dlh['Date'].dt.dayofweek/7)
    dlh['day_cos'] = np.cos(2*np.pi*dlh['Date'].dt.dayofweek/7)
    dlh['month_sin'] = np.sin(2*np.pi*(dlh['Date'].dt.month-1)/12)
    dlh['month_cos'] = np.cos(2*np.pi*(dlh['Date'].dt.month-1)/12)
    dlh['Is_Weekend'] = dlh['Date'].dt.dayofweek.isin([5,6]).astype(int)
    dlh['hour_sin'] = np.sin(2*np.pi*(dlh['Date'].dt.hour-8)/9)
    dlh['hour_cos'] = np.cos(2*np.pi*(dlh['Date'].dt.hour-8)/9)
    time_feats = ['day_sin','day_cos','month_sin','month_cos','Is_Weekend','hour_sin','hour_cos']

    # Merge daily exogenous (broadcast to each hour of that day)
    dlh['Date_Day'] = dlh['Date'].dt.normalize()
    dlh = dlh.merge(dw.rename(columns={'Date':'Date_Day'}), on='Date_Day', how='left')
    dlh = dlh.merge(dh.rename(columns={'Date':'Date_Day'}), on='Date_Day', how='left')
    dlh = dlh.merge(de.rename(columns={'Date':'Date_Day'}), on='Date_Day', how='left')
    dlh = dlh.drop(columns=['Date_Day'])
    for c in dlh.select_dtypes(include=[np.number]).columns: dlh[c]=dlh[c].fillna(0)

    exclude = ['Date','Sales','Product_Name','Time','date']
    base_feat_cols = [c for c in dw.columns.tolist()+dh.columns.tolist()+de.columns.tolist()
                      if c not in ['Date','Time','date'] and c in dlh.columns]
    base_feat_cols = list(dict.fromkeys(base_feat_cols))  # dedupe preserving order
    base_feat_cols = [c for c in base_feat_cols if dlh[c].dtype.kind in 'iufc']

    dlh = dlh.sort_values(['Product_Name','Date']).reset_index(drop=True)
    enc = LabelEncoder()
    dlh['Product_ID'] = enc.fit_transform(dlh['Product_Name'])

    # Hourly lags
    dlh['sales_lag_1h'] = dlh.groupby('Product_Name')['Sales'].shift(1)
    dlh['sales_lag_9h'] = dlh.groupby('Product_Name')['Sales'].shift(9)
    dlh['sales_lag_63h'] = dlh.groupby('Product_Name')['Sales'].shift(63)
    dlh['sales_rolling_9h_mean'] = dlh.groupby('Product_Name')['Sales'].shift(1).groupby(
        dlh['Product_Name']).transform(lambda x: x.rolling(9,min_periods=1).mean())
    dlh['sales_rolling_9h_std'] = dlh.groupby('Product_Name')['Sales'].shift(1).groupby(
        dlh['Product_Name']).transform(lambda x: x.rolling(9,min_periods=1).std()).fillna(0)

    lag_feats = ['sales_lag_1h','sales_lag_9h','sales_lag_63h','sales_rolling_9h_mean','sales_rolling_9h_std']
    dlh = dlh.dropna(subset=lag_feats).reset_index(drop=True)
    feature_cols = base_feat_cols + time_feats + ['Product_ID'] + lag_feats
    # Dedupe
    feature_cols = list(dict.fromkeys(feature_cols))
    feature_cols = [c for c in feature_cols if c in dlh.columns]

    max_date = dlh['Date'].max()
    val_start = max_date - pd.Timedelta(days=60)
    val_end_dt = max_date - pd.Timedelta(days=30)

    feat_scaler = MinMaxScaler(); tgt_scaler = MinMaxScaler()
    train_data = dlh[dlh['Date'] <= val_start]
    feat_scaler.fit(train_data[feature_cols])
    tgt_scaler.fit(train_data[['Sales']])

    def build_seqs(pdf):
        if len(pdf) < SEQ_LEN+1: return None, None, None
        sf = feat_scaler.transform(pdf[feature_cols])
        st = tgt_scaler.transform(pdf[['Sales']]).flatten()
        X, y, d = [], [], []
        for i in range(len(sf)-SEQ_LEN):
            X.append(sf[i:i+SEQ_LEN]); y.append(st[i+SEQ_LEN]); d.append(pdf['Date'].values[i+SEQ_LEN])
        return np.array(X), np.array(y), pd.to_datetime(d)

    all_Xt, all_yt, all_Xv, all_yv = [], [], [], []
    for pn in dlh['Product_Name'].unique():
        pdf = dlh[dlh['Product_Name']==pn].sort_values('Date')
        res = build_seqs(pdf)
        if res[0] is None: continue
        X, y, dates = res
        tm = dates <= np.datetime64(val_start)
        vm = (dates > np.datetime64(val_start)) & (dates <= np.datetime64(val_end_dt))
        if tm.sum()>0: all_Xt.append(X[tm]); all_yt.append(y[tm])
        if vm.sum()>0: all_Xv.append(X[vm]); all_yv.append(y[vm])

    X_train = np.vstack(all_Xt); y_train = np.concatenate(all_yt)
    X_val = np.vstack(all_Xv); y_val = np.concatenate(all_yv)
    shuf = np.random.permutation(len(X_train))
    X_train, y_train = X_train[shuf], y_train[shuf]

    print(f"    Sequences: {X_train.shape[0]} train, {X_val.shape[0]} val")
    print(f"    Tuning LSTM Hourly ({OPTUNA_LSTM_TRIALS} trials, CPU)...")

    def objective(trial):
        tf.keras.backend.clear_session()
        bs = trial.suggest_categorical('batch_size',[64,128,256])
        u1 = trial.suggest_int('lstm_1_units',64,192,step=32)
        u2 = trial.suggest_int('lstm_2_units',32,96,step=16)
        dr = trial.suggest_float('dropout_rate',0.1,0.4)
        lr = trial.suggest_float('learning_rate',1e-4,1e-2,log=True)
        l2r = trial.suggest_float('l2_reg',1e-5,1e-2,log=True)
        m = Sequential([Input(shape=(SEQ_LEN,len(feature_cols))),
            LSTM(u1,activation='tanh',return_sequences=True), Dropout(dr),
            LSTM(u2,activation='tanh'), Dropout(dr),
            Dense(64,activation='relu',kernel_regularizer=l2(l2r)),
            Dense(1)])
        m.compile(optimizer=Adam(learning_rate=lr), loss=Huber())
        m.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=30, batch_size=bs,
              callbacks=[EarlyStopping(patience=6,restore_best_weights=True)], verbose=0)
        vp = tgt_scaler.inverse_transform(m.predict(X_val,verbose=0)).flatten()
        va = tgt_scaler.inverse_transform(y_val.reshape(-1,1)).flatten()
        return np.sqrt(np.mean((va-vp)**2))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=OPTUNA_LSTM_TRIALS)
    bp = study.best_params
    print(f"    Best RMSE: {study.best_value:.4f}. Training final model...")

    tf.keras.backend.clear_session()
    final = Sequential([Input(shape=(SEQ_LEN,len(feature_cols))),
        LSTM(bp['lstm_1_units'],activation='tanh',return_sequences=True), Dropout(bp['dropout_rate']),
        LSTM(bp['lstm_2_units'],activation='tanh'), Dropout(bp['dropout_rate']),
        Dense(64,activation='relu',kernel_regularizer=l2(bp['l2_reg'])),
        Dense(1)])
    final.compile(optimizer=Adam(learning_rate=bp['learning_rate']), loss=Huber())
    final.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=100,
              batch_size=bp['batch_size'],
              callbacks=[EarlyStopping(patience=12,restore_best_weights=True),
                         ReduceLROnPlateau(factor=0.5,patience=5,min_lr=1e-6)], verbose=0)

    # Recursive hourly forecast → aggregate to daily
    print(f"    Forecasting {forecast_days} days recursively (hourly)...")
    lag_idx = [feature_cols.index(f) for f in lag_feats]
    lag_min = feat_scaler.data_min_[lag_idx]
    lag_range = feat_scaler.data_range_[lag_idx]
    lag_range[lag_range==0] = 1.0

    results = []
    for pn in sorted(dlh['Product_Name'].unique()):
        pdf = dlh[dlh['Product_Name']==pn].sort_values('Date')
        if len(pdf) < SEQ_LEN: continue
        tail = pdf.tail(SEQ_LEN)
        seq = feat_scaler.transform(tail[feature_cols])
        sales_hist = list(pdf['Sales'].values[-max(SEQ_LEN,63):])
        max_dt = pdf['Date'].max()

        forecast_hours_list = []
        for d in range(forecast_days):
            day = max_dt.normalize() + pd.Timedelta(days=d+1)
            for h in BUSINESS_HOURS:
                forecast_hours_list.append(day + pd.Timedelta(hours=h))

        for hi, cur_dt in enumerate(forecast_hours_list):
            pred_s = final.predict(seq.reshape(1,SEQ_LEN,-1), verbose=0)[0,0]
            pred_r = max(0, tgt_scaler.inverse_transform([[pred_s]])[0,0])
            sales_hist.append(pred_r)
            results.append({'Date':cur_dt,'Product_Name':pn,'Forecast_h':pred_r})

            if hi+1 < len(forecast_hours_list):
                next_feat = seq[-1].copy()
                # Recompute hourly lags
                h = sales_hist
                raw = np.array([h[-1], h[-9] if len(h)>=9 else h[0],
                    h[-63] if len(h)>=63 else h[0],
                    np.mean(h[-9:]) if len(h)>=9 else np.mean(h),
                    np.std(h[-9:]) if len(h)>=9 else 0.0])
                scaled = (raw-lag_min)/lag_range
                for li,fi in enumerate(lag_idx): next_feat[fi]=scaled[li]
                # Update time features for next hour
                ndt = forecast_hours_list[hi+1]
                dow=ndt.dayofweek; mon=ndt.month; hr=ndt.hour
                time_vals = {'day_sin':np.sin(2*np.pi*dow/7),'day_cos':np.cos(2*np.pi*dow/7),
                    'month_sin':np.sin(2*np.pi*(mon-1)/12),'month_cos':np.cos(2*np.pi*(mon-1)/12),
                    'Is_Weekend':1 if dow>=5 else 0,
                    'hour_sin':np.sin(2*np.pi*(hr-8)/9),'hour_cos':np.cos(2*np.pi*(hr-8)/9)}
                for tf_name, tv in time_vals.items():
                    if tf_name in feature_cols:
                        fi=feature_cols.index(tf_name)
                        next_feat[fi]=(tv-feat_scaler.data_min_[fi])/(feat_scaler.data_range_[fi]+1e-8)
                seq = np.roll(seq,-1,axis=0); seq[-1]=next_feat

    tf.keras.backend.clear_session()

    # Rollup hourly → daily
    rdf = pd.DataFrame(results)
    rdf['Date_Day'] = rdf['Date'].dt.normalize()
    daily = rdf.groupby(['Date_Day','Product_Name'])['Forecast_h'].sum().reset_index()
    daily = daily.rename(columns={'Date_Day':'Date','Forecast_h':'Forecast'})
    daily['Forecast'] = daily['Forecast'].round().astype(int)
    return daily[['Date','Product_Name','Forecast']]


# ── RUNNER REGISTRY ──
MODEL_RUNNERS = {
    'xgb_simple_daily':   run_xgb_simple_daily,
    'xgb_improved_daily': run_xgb_improved_daily,
    'xgb_simple_hourly':  run_xgb_simple_hourly,
    'xgb_improved_hourly':run_xgb_improved_hourly,
    'arima':              run_arima,
    'prophet_daily':      run_prophet_daily,
    'prophet_hourly':     run_prophet_hourly,
    'lstm_daily':         run_lstm_daily,
    'lstm_hourly':        run_lstm_hourly,
}


# ══════════════════════════════════════════════════════════════
# PDF REPORT (same as v2 — generate_report function)
# ══════════════════════════════════════════════════════════════
# [The generate_report function is identical to v2 — it's imported/reused.
#  For brevity in this file, I'm including it inline below.]

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

    def make_table(data,cw,hc=hdr_bg):
        t=Table(data,colWidths=cw,repeatRows=1)
        s=[('BACKGROUND',(0,0),(-1,0),hc),('TEXTCOLOR',(0,0),(-1,0),hdr_fg),
           ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,0),8),
           ('FONTSIZE',(0,1),(-1,-1),8),('ALIGN',(1,0),(-1,-1),'CENTER'),
           ('ALIGN',(0,0),(0,-1),'LEFT'),('GRID',(0,0),(-1,-1),0.3,grid),
           ('BOTTOMPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),3)]
        for i in range(1,len(data)):
            if i%2==0: s.append(('BACKGROUND',(0,i),(-1,i),alt))
        t.setStyle(TableStyle(s)); return t

    def add_two_col(items,c1,c2,hc=ingr_hdr):
        half=(len(items)+1)//2; l,r=items[:half],items[half:]
        data=[[c1,c2,'',c1,c2]]
        for i in range(half):
            ln=l[i][0] if i<len(l) else ''; lv=l[i][1] if i<len(l) else ''
            rn=r[i][0] if i<len(r) else ''; rv=r[i][1] if i<len(r) else ''
            data.append([ln,lv,'',rn,rv])
        t=Table(data,colWidths=[4*cm,2.5*cm,0.4*cm,4*cm,2.5*cm])
        s=[('BACKGROUND',(0,0),(1,0),hc),('BACKGROUND',(3,0),(4,0),hc),
           ('TEXTCOLOR',(0,0),(-1,0),hdr_fg),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
           ('FONTSIZE',(0,0),(-1,-1),8),('ALIGN',(1,0),(1,-1),'RIGHT'),('ALIGN',(4,0),(4,-1),'RIGHT'),
           ('GRID',(0,0),(1,-1),0.3,grid),('GRID',(3,0),(4,-1),0.3,grid),
           ('BOTTOMPADDING',(0,0),(-1,-1),2),('TOPPADDING',(0,0),(-1,-1),2)]
        for i in range(1,len(data)):
            if i%2==0: s.append(('BACKGROUND',(0,i),(1,i),alt)); s.append(('BACKGROUND',(3,i),(4,i),alt))
        t.setStyle(TableStyle(s)); return t

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

    if horizon=='day':
        for d in forecast_dates:
            dn=DAY_NAMES[d.dayofweek]
            story.append(Paragraph(f"Kitchen Prep — {dn} {d.strftime('%d %B %Y')}",sec_s))
            story.append(Paragraph("Forecast quantities for each menu item, sorted by expected volume. Prepare these before service.",note_s))
            story.append(Paragraph("Items to Prepare",subsec_s))
            dp=forecast_df[(forecast_df['Date']==d)&(forecast_df['Forecast']>0)].sort_values('Forecast',ascending=False)
            items=[(str(r['Product_Name']),str(int(r['Forecast']))) for _,r in dp.iterrows()]
            if items: story.append(add_two_col(items,'Item','Qty',hdr_bg)); story.append(Paragraph(f"Day total: {int(dp['Forecast'].sum())} items",body_s))
            story.append(Spacer(1,4*mm))
            story.append(Paragraph("Ingredients Required",subsec_s))
            story.append(Paragraph("Total raw ingredients for today. Quantities over 500g shown in kg. Check stock and defrost accordingly.",note_s))
            di=daily_ingredients[(daily_ingredients['Date']==d)&(daily_ingredients['total_qty']>0)]
            di=convert_grams_to_kg(di).sort_values('total_qty',ascending=False)
            ingr=[(str(r['ingredient']),fmt_qty(r['total_qty'],r['unit'])) for _,r in di.iterrows()]
            if ingr: story.append(add_two_col(ingr,'Ingredient','Quantity'))
            if d!=forecast_dates[-1]: story.append(PageBreak())

    elif horizon=='week':
        story.append(Paragraph("Weekly Product Overview",sec_s))
        story.append(Paragraph("Forecast per product per day. A dash means zero. Use for daily prep planning and staffing.",note_s))
        pivot=forecast_df.pivot_table(index='Product_Name',columns='Date',values='Forecast',aggfunc='sum',fill_value=0)
        pivot['TOTAL']=pivot.sum(axis=1); pivot=pivot[pivot['TOTAL']>0].sort_values('TOTAL',ascending=False)
        hdrs=['Product']+[f"{DAY_NAMES[d.dayofweek][:3]}\n{d.day}" for d in forecast_dates]+['Total']
        td=[hdrs]
        for pn,row in pivot.iterrows():
            r=[str(pn)]+[str(int(row.get(d,0))) if int(row.get(d,0))>0 else '-' for d in forecast_dates]+[str(int(row['TOTAL']))]
            td.append(r)
        td.append(['TOTAL']+[str(int(pivot[d].sum())) for d in forecast_dates]+[str(int(pivot['TOTAL'].sum()))])
        nc=len(hdrs); cw=[3.8*cm]+[1.6*cm]*(nc-2)+[1.8*cm]
        tw=sum(cw)
        if tw>W: cw=[w*W/tw for w in cw]
        t=Table(td,colWidths=cw,repeatRows=1)
        tsl=[('BACKGROUND',(0,0),(-1,0),hdr_bg),('TEXTCOLOR',(0,0),(-1,0),hdr_fg),
             ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),7),
             ('ALIGN',(1,0),(-1,-1),'CENTER'),('ALIGN',(0,0),(0,-1),'LEFT'),('GRID',(0,0),(-1,-1),0.3,grid),
             ('BOTTOMPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),3),
             ('BACKGROUND',(0,-1),(-1,-1),total_bg),('FONTNAME',(0,-1),(-1,-1),'Helvetica-Bold')]
        for i in range(1,len(td)-1):
            if i%2==0: tsl.append(('BACKGROUND',(0,i),(-1,i),alt))
        t.setStyle(TableStyle(tsl)); story.append(t); story.append(PageBreak())
        story.append(Paragraph("Supplier Order List — Weekly Totals",sec_s))
        story.append(Paragraph("Total ingredients for the week, sorted by volume. Quantities over 500g shown in kg. Consider a 10-15% buffer for popular items.",note_s))
        wi=(daily_ingredients.groupby(['ingredient','unit'])['total_qty'].sum().reset_index().sort_values('total_qty',ascending=False))
        wi=wi[wi['total_qty']>0]; wi=convert_grams_to_kg(wi)
        ingr=[(str(r['ingredient']),fmt_qty(r['total_qty'],r['unit'])) for _,r in wi.iterrows()]
        if ingr: story.append(add_two_col(ingr,'Ingredient','Order Qty'))

    elif horizon=='month':
        story.append(Paragraph("Weekly Breakdown",sec_s))
        story.append(Paragraph("Monthly forecast split by calendar week for staffing and inventory planning.",note_s))
        forecast_df=forecast_df.copy(); forecast_df['week']=forecast_df['Date'].dt.isocalendar().week.astype(int)
        wt=forecast_df.groupby('week')['Forecast'].sum().reset_index()
        wtd=[['Week','Total','Daily Avg']]
        for _,wr in wt.iterrows():
            wk=int(wr['week']); tot=int(wr['Forecast']); diw=len(forecast_df[forecast_df['week']==wk]['Date'].unique())
            wtd.append([f"Week {wk}",str(tot),str(int(tot/diw) if diw>0 else 0)])
        story.append(make_table(wtd,[3*cm,4*cm,4*cm])); story.append(Spacer(1,6*mm))
        story.append(Paragraph("Top 15 Products by Volume",sec_s))
        story.append(Paragraph("Highest-demand products. Ensure reliable supply and sufficient prep capacity.",note_s))
        top=forecast_df.groupby('Product_Name')['Forecast'].sum().sort_values(ascending=False).head(15).reset_index()
        td=[['#','Product','Monthly Total','Daily Avg']]
        for i,(_,r) in enumerate(top.iterrows(),1): td.append([str(i),str(r['Product_Name']),str(int(r['Forecast'])),f"{r['Forecast']/n_days:.1f}"])
        story.append(make_table(td,[1*cm,5*cm,3*cm,3*cm])); story.append(Spacer(1,6*mm))
        story.append(Paragraph("Lowest Volume Products",sec_s))
        story.append(Paragraph("Consider whether these items justify their menu space, prep time, and ingredient costs.",note_s))
        bot=forecast_df.groupby('Product_Name')['Forecast'].sum().sort_values().head(10).reset_index()
        bd=[['Product','Monthly Total','Daily Avg']]
        for _,r in bot.iterrows(): bd.append([str(r['Product_Name']),str(int(r['Forecast'])),f"{r['Forecast']/n_days:.1f}"])
        story.append(make_table(bd,[5*cm,3*cm,3*cm])); story.append(PageBreak())
        story.append(Paragraph("Day-of-Week Demand Patterns",sec_s))
        story.append(Paragraph("Average daily volume by day of week. Plan staffing rotas and prep schedules accordingly.",note_s))
        forecast_df['dow_name']=forecast_df['Date'].dt.dayofweek.map({i:DAY_NAMES[i] for i in range(7)})
        da=forecast_df.groupby('dow_name')['Forecast'].sum(); dc=forecast_df.groupby('dow_name')['Date'].nunique()
        dd=[['Day','Total','Daily Avg']]
        for dn in DAY_NAMES:
            if dn in da.index: dd.append([dn,str(int(da[dn])),str(int(da[dn]/dc[dn]))])
        story.append(make_table(dd,[4*cm,3*cm,3*cm])); story.append(Spacer(1,6*mm))
        story.append(Paragraph("Monthly Ingredient Requirements",sec_s))
        story.append(Paragraph("Total ingredients for the month. Quantities over 500g shown in kg. Use for bulk ordering and supplier negotiations.",note_s))
        wi=(daily_ingredients.groupby(['ingredient','unit'])['total_qty'].sum().reset_index().sort_values('total_qty',ascending=False))
        wi=wi[wi['total_qty']>0]; wi=convert_grams_to_kg(wi)
        ingr=[(str(r['ingredient']),fmt_qty(r['total_qty'],r['unit'])) for _,r in wi.iterrows()]
        if ingr: story.append(add_two_col(ingr,'Ingredient','Monthly Total'))

    doc.build(story); print(f"    PDF saved: {output_path}")


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Multi-Model Forecast Pipeline v3')
    parser.add_argument('--model', help='Force a specific runner (xgb_improved_daily, arima, prophet_daily, etc.)')
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

    configs = [('day',1,'Daily Kitchen Prep Sheet'),('week',7,'Weekly Ordering Guide'),('month',30,'Monthly Stakeholder Report')]

    # Group by runner to avoid retraining
    runner_to_horizons = {}
    for hk,hd,ht in configs:
        rn = runners[hk]
        if rn not in runner_to_horizons: runner_to_horizons[rn]=[]
        runner_to_horizons[rn].append((hk,hd,ht))

    forecasts = {}
    for rn,hl in runner_to_horizons.items():
        md=max(hd for _,hd,_ in hl)
        print(f"\n  ── Running {rn.upper()} (forecasting {md} days) ──")
        fn = MODEL_RUNNERS.get(rn)
        if not fn:
            print(f"    ERROR: No runner '{rn}'. Skipping.")
            continue
        forecasts[rn] = fn(md)
        print(f"    Done: {len(forecasts[rn])} predictions")

    print(f"\n  ── Generating Reports ──")
    paths = []
    for hk,hd,ht in configs:
        rn = runners[hk]
        if rn not in forecasts: continue
        ff = forecasts[rn]
        dates = sorted(ff['Date'].unique())[:hd]
        fdf = ff[ff['Date'].isin(dates)].copy()
        merged = fdf.merge(recipes,left_on='Product_Name',right_on='product',how='left')
        merged['total_qty'] = merged['Forecast'] * merged['quantity']
        di = merged.groupby(['Date','ingredient','unit'])['total_qty'].sum().reset_index().sort_values(['Date','ingredient'])
        mn = best_models[hk]['model_type'] if best_models[hk] else rn.upper()
        mi = best_models[hk]
        fn = f"forecast_{hk}_{fdf['Date'].min().strftime('%Y-%m-%d')}.pdf"
        op = os.path.join(RESULTS_DIR,fn)
        print(f"    {ht} ({mn})...")
        generate_report(fdf,di,mn,mi,hk,op)
        paths.append(op)

    if forecasts:
        longest=max(forecasts.values(),key=len)
        cp=os.path.join(RESULTS_DIR,f"forecast_all_{datetime.now().strftime('%Y%m%d')}.csv")
        longest.to_csv(cp,index=False)
        print(f"\n    CSV: {cp}")

    print("\n"+"═"*58)
    print("  PIPELINE COMPLETE — 3 reports generated")
    for p in paths: print(f"    → {p}")
    print("═"*58)


if __name__ == '__main__':
    main()
