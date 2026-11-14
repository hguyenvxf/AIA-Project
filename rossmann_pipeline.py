"""
ME3297 team project - Rossmann store sales (dataset no. 5)
==========================================================
Decision supported : daily sales forecast per store, 6 weeks (42 days) ahead,
                     for staffing and replenishment planning.
Target             : Sales (EUR per store per open day).
Known at decision  : calendar, store attributes, planned Promo / Promo2,
                     school/state holidays, and sales history that is at
                     least 42 days old.  NOT known: Customers (dropped).

HOW TO RUN
----------
1. Download train.csv and store.csv from
   https://www.kaggle.com/competitions/rossmann-store-sales/data
   (free Kaggle account + accept the competition rules once).
2. Nothing to configure: the script searches your computer (script folder,
   Downloads, Desktop, Documents, OneDrive) for the Rossmann train.csv and
   store.csv and checks their column names.  To force a folder, set the
   environment variable ROSSMANN_DIR.
3. pip install pandas numpy scikit-learn
4. python rossmann_pipeline.py

Do NOT upload the raw data to BKEL (project brief, section 7).
Set FAST_MODE = True below to debug quickly on a small sample.
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42
HORIZON_DAYS = 42        # forecast horizon = 6 weeks; lags must be >= this old
TEST_DAYS = 42           # the test set is the LAST 42 days of the file
N_FOLDS = 4              # rolling-origin validation folds
VAL_DAYS = 42            # length of each validation window
MAX_FIT_ROWS = 150_000   # rows used to fit each model (speed); None = all
FAST_MODE = False        # True = tiny run for debugging

if FAST_MODE:
    MAX_FIT_ROWS = 20_000

rng = np.random.default_rng(SEED)


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ----------------------------------------------------------------------------
# STEP 1 - LOAD AND DOCUMENT THE DATA
# ----------------------------------------------------------------------------
section("1. LOAD AND DOCUMENT THE DATA")

def is_rossmann_folder(folder):
    """True if folder holds the Rossmann train.csv AND store.csv (checks the headers)."""
    try:
        t = pd.read_csv(os.path.join(folder, "train.csv"), nrows=0).columns
        s_ = pd.read_csv(os.path.join(folder, "store.csv"), nrows=0).columns
        return {"Store", "Sales", "Customers", "Date"} <= set(t) and "StoreType" in set(s_)
    except Exception:
        return False


def find_data_dir():
    """Search this computer for the Rossmann train.csv + store.csv.

    Order: ROSSMANN_DIR variable -> script folder -> working folder ->
    a bounded scan (4 levels deep) of Downloads, Desktop, Documents, OneDrive
    and the whole user folder.
    """
    home = os.path.expanduser("~")
    quick = [
        os.environ.get("ROSSMANN_DIR"),
        SCRIPT_DIR, os.path.join(SCRIPT_DIR, "data"),
        os.getcwd(), os.path.join(os.getcwd(), "data"),
        os.path.join(home, "Downloads"),
    ]
    for folder in quick:
        if folder and os.path.isdir(folder) and is_rossmann_folder(folder):
            return folder

    print("Searching your computer for train.csv and store.csv ...")
    skip = {"site-packages", "node_modules", "appdata", "$recycle.bin", ".venv", "venv"}
    roots = [os.path.join(home, d) for d in ("Downloads", "Desktop", "Documents", "OneDrive")]
    roots = [r for r in roots if os.path.isdir(r)] + [home]
    seen = set()
    for root in roots:
        base_depth = root.rstrip(os.sep).count(os.sep)
        for current, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d.lower() not in skip and not d.startswith(".")]
            if current.count(os.sep) - base_depth >= 4:
                dirs[:] = []
            if current in seen:
                continue
            seen.add(current)
            if "train.csv" in files and "store.csv" in files and is_rossmann_folder(current):
                return current
    sys.exit("Could not find the Rossmann train.csv and store.csv on this computer.\n"
             "Download them from Kaggle (see HOW TO RUN), then either put them next to "
             "this script or set the ROSSMANN_DIR environment variable.")


DATA_DIR = r'C:\WorkSpace\HCMUT\Semester_7\TRÍ TUỆ NHÂN TẠO ỨNG DỤNG\project\data\raw'
print("Using data folder:", DATA_DIR)
train_path = os.path.join(DATA_DIR, "train.csv")
store_path = os.path.join(DATA_DIR, "store.csv")

train_df = pd.read_csv(train_path, low_memory=False)
store_df = pd.read_csv(store_path)
print("train.csv:", train_df.shape, "| store.csv:", store_df.shape)

df = train_df.merge(store_df, on="Store", how="left")
df["Date"] = pd.to_datetime(df["Date"])
df["StateHoliday"] = df["StateHoliday"].astype(str)   # mixes 0 and '0'
print("Date range:", df["Date"].min().date(), "->", df["Date"].max().date())
print("Stores:", df["Store"].nunique())

if FAST_MODE:
    all_stores = df["Store"].unique()
    keep_stores = rng.choice(all_stores, size=min(60, len(all_stores)), replace=False)
    df = df[df["Store"].isin(keep_stores)].copy()
    print("FAST_MODE: kept", len(keep_stores), "stores ->", df.shape)

# ----------------------------------------------------------------------------
# STEP 2 - DATA QUALITY REPORT (validation rules, duplicates, missing, outliers)
# ----------------------------------------------------------------------------
section("2. DATA QUALITY REPORT")

n0 = len(df)
print("Rows:", n0, "| Columns:", df.shape[1])

print("\n-- Missing cells per column (only columns with gaps) --")
miss = df.isna().sum()
print(miss[miss > 0].to_string() if (miss > 0).any() else "none")

print("\n-- Validation rules (count of rows that break each rule) --")
dup_key = int(df.duplicated(subset=["Store", "Date"]).sum())
neg_sales = int((df["Sales"] < 0).sum())
open_zero = int(((df["Open"] == 1) & (df["Sales"] == 0)).sum())
closed_sales = int(((df["Open"] == 0) & (df["Sales"] > 0)).sum())
print(f"duplicate (Store, Date) keys            : {dup_key}")
print(f"negative Sales                          : {neg_sales}")
print(f"store OPEN but Sales == 0               : {open_zero}")
print(f"store CLOSED but Sales > 0              : {closed_sales}")
print(f"closed days (Open == 0)                 : {int((df['Open'] == 0).sum())}")

print("\n-- Target distribution on open days --")
open_sales = df.loc[(df["Open"] == 1) & (df["Sales"] > 0), "Sales"]
print(open_sales.describe().round(1).to_string())
print("skewness:", round(open_sales.skew(), 2),
      "| after log1p:", round(np.log1p(open_sales).skew(), 2))
q1, q3 = open_sales.quantile([0.25, 0.75])
iqr = q3 - q1
hi = q3 + 1.5 * iqr
print(f"Tukey upper fence: {hi:,.0f} -> {(open_sales > hi).sum()} rows flagged "
      "(extreme but valid: promotions / December peaks, so KEPT)")

# Cleaning decisions (documented, not fitted on any data):
df = df.drop_duplicates(subset=["Store", "Date"])
df = df[df["Sales"] >= 0]
print("\nDecisions: drop duplicate keys; drop negative sales; closed days are")
print("excluded from modelling (sales are trivially 0); open days with 0 sales")
print("are treated as recording errors and dropped AFTER lag features are built.")

# ----------------------------------------------------------------------------
# STEP 3 - FEATURES (calendar, store attributes, lags that respect the horizon)
# ----------------------------------------------------------------------------
section("3. FEATURE ENGINEERING")

# 3a. Calendar features (Week 2 / Week 3: features from a date column)
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

# 3b. Deterministic (row-wise) features - nothing is fitted, so no leakage.
# Months since a competitor opened; NaN means unknown / none -> 0 plus a flag.
df["comp_unknown"] = df["CompetitionOpenSinceYear"].isna().astype(int)
df["comp_open_months"] = (
    (df["Year"] - df["CompetitionOpenSinceYear"]) * 12
    + (df["Month"] - df["CompetitionOpenSinceMonth"])
).clip(lower=0).fillna(0)

# Weeks since the store joined Promo2; a blank genuinely means "none" -> 0.
df["promo2_weeks"] = np.where(
    df["Promo2"] == 1,
    (df["Year"] - df["Promo2SinceYear"]) * 52 + (df["WeekOfYear"] - df["Promo2SinceWeek"]),
    0,
)
df["promo2_weeks"] = pd.Series(df["promo2_weeks"], index=df.index).clip(lower=0).fillna(0)

# 3c. Lag features built from PAST sales only, at least HORIZON_DAYS old.
# Pivot to a complete store x calendar-day grid so shift(42) means 42 DAYS.
df["Sales_open"] = df["Sales"].where((df["Open"] == 1) & (df["Sales"] > 0))
wide = df.pivot_table(index="Date", columns="Store", values="Sales_open")
wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max()))
wide.index.name = "Date"

past = wide.shift(HORIZON_DAYS)                       # known 6 weeks ahead
lag_tables = {
    "lag_42": past,                                   # same weekday, 6 weeks ago
    "roll_mean_28": past.rolling(28, min_periods=5).mean(),
    "roll_mean_91": past.rolling(91, min_periods=15).mean(),
}
for name, table in lag_tables.items():
    long = table.reset_index().melt(id_vars="Date", var_name="Store", value_name=name)
    long["Store"] = long["Store"].astype(df["Store"].dtype)
    df = df.merge(long, on=["Date", "Store"], how="left")
print("Lag features built:", list(lag_tables))
print("Leakage control: every lag is shifted by", HORIZON_DAYS,
      "days, so only sales known at forecast time are used.")

# Model rows: open days with positive sales
before = len(df)
df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()
print(f"Rows kept for modelling: {len(df):,} (removed {before - len(df):,})")

# --- THÊM MỚI: lưu dữ liệu đã xử lý để 02_models.py dùng -------------------
df.to_pickle("rossmann_clean.pkl")
print("Đã lưu dữ liệu đã xử lý -> rossmann_clean.pkl", df.shape)

ONLY_PREPARE_DATA = True   # True = dừng ở đây; phần mô hình làm trong 02_models.py
if ONLY_PREPARE_DATA:
    sys.exit(0)
# ---------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# STEP 4 - SPLIT BY DATE (before any fitting)
# ----------------------------------------------------------------------------
section("4. TIME-BASED SPLIT")

df = df.sort_values(["Date", "Store"]).reset_index(drop=True)
last_date = df["Date"].max()
test_start = last_date - pd.Timedelta(days=TEST_DAYS - 1)
train_mask = df["Date"] < test_start
train_set = df[train_mask].copy()
test_set = df[~train_mask].copy()
print(f"Train: {train_set['Date'].min().date()} -> {train_set['Date'].max().date()} "
      f"({len(train_set):,} rows)")
print(f"Test : {test_set['Date'].min().date()} -> {test_set['Date'].max().date()} "
      f"({len(test_set):,} rows)  <- opened once, at the end")

NUM_FEATURES = [
    "CompetitionDistance", "comp_open_months", "comp_unknown",
    "Promo", "Promo2", "promo2_weeks", "SchoolHoliday",
    "DayOfWeek", "Month", "Day", "WeekOfYear",
    "lag_42", "roll_mean_28", "roll_mean_91",
]
CAT_FEATURES = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES
# Year is deliberately NOT a feature: the test year lies outside the
# training range and a tree cannot extrapolate.  Customers is dropped
# (not known in advance).  Store ID is not used as a raw number.

X_train, y_train = train_set[ALL_FEATURES], train_set["Sales"]
X_test, y_test = test_set[ALL_FEATURES], test_set["Sales"]


# ----------------------------------------------------------------------------
# HELPERS - preprocessing, metrics, validation folds
# ----------------------------------------------------------------------------
def make_preprocessor():
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, NUM_FEATURES),
        ("cat", cat_pipe, CAT_FEATURES),
    ])


def make_model(regressor):
    """Preprocessing + regressor trained on log1p(Sales) (Sales is right-skewed)."""
    pipe = Pipeline([("prep", make_preprocessor()), ("model", regressor)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    err = y_true - y_pred
    mask = y_true > 0
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "RMSPE": float(np.sqrt(np.mean((err[mask] / y_true[mask]) ** 2))),
        "Bias": float(np.mean(-err)),   # >0 means over-forecasting
    }


def baseline_predict(X, fallback):
    """Baseline = same weekday 6 weeks ago; else 28-day mean; else train median."""
    p = X["lag_42"].copy()
    p = p.fillna(X["roll_mean_28"]).fillna(X["roll_mean_91"]).fillna(fallback)
    return p.to_numpy()


def rolling_origin_folds(dates, n_folds, val_days):
    """Each fold validates on a later 42-day window; training is everything before."""
    last = dates.max()
    folds = []
    for i in range(n_folds, 0, -1):
        val_end = last - pd.Timedelta(days=(i - 1) * val_days)
        val_start = val_end - pd.Timedelta(days=val_days - 1)
        tr_idx = np.where(dates < val_start)[0]
        va_idx = np.where((dates >= val_start) & (dates <= val_end))[0]
        folds.append((tr_idx, va_idx))
    return folds


def sample_rows(idx, max_rows):
    if max_rows is None or len(idx) <= max_rows:
        return idx
    return np.sort(rng.choice(idx, size=max_rows, replace=False))


folds = rolling_origin_folds(train_set["Date"].reset_index(drop=True), N_FOLDS, VAL_DAYS)
train_median = float(y_train.median())


