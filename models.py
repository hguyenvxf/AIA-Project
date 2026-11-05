"""
ME3297 - Rossmann - FILE 2: MÁY HỌC
====================================
Chạy SAU rossmann_pipeline.py (file đó xử lý dữ liệu và lưu rossmann_clean.pkl).
Lệnh: python 02_models.py

Các bước:
  B1. Đọc dữ liệu đã xử lý, chọn đặc trưng
  B2. Chia train / test theo thời gian  (test = 42 ngày cuối, chỉ mở 1 lần ở B8)
  B3. Tạo các fold kiểm định rolling-origin trên tập train
  B4. Chọn chỉ số đánh giá  (MAE chính, RMSPE và Bias phụ)
  B5. Pipeline tái lập được  (tiền xử lý + mô hình trong 1 khối)
  B6. Mô hình baseline
  B7. Tinh chỉnh siêu tham số + kiểm tra overfitting cho 3 họ mô hình
      -> so sánh với baseline -> CHỌN mô hình tốt nhất theo kết quả kiểm định
  B8. Đánh giá trên tập test (lần duy nhất)
  B9. Diễn giải mô hình

Kết quả: in ra màn hình, bảng .csv và hình .png trong thư mục results/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # vẽ hình ra file, không cần mở cửa sổ
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.inspection import permutation_importance

# ---------------------------------------------------------------------------
# CÀI ĐẶT CHUNG
# ---------------------------------------------------------------------------
SEED = 42            # cố định ngẫu nhiên -> chạy lại ra đúng số cũ
TEST_DAYS = 42       # tập test = 42 ngày cuối
VAL_DAYS = 42        # mỗi fold kiểm định dài 42 ngày (= tầm dự báo)
N_FOLDS = 3          # số fold rolling-origin
FIT_ROWS = 100_000   # số dòng dùng để huấn luyện mỗi lần (cho nhanh)

os.makedirs("results", exist_ok=True)
rng = np.random.default_rng(SEED)

# ===========================================================================
# B1. ĐỌC DỮ LIỆU ĐÃ XỬ LÝ, CHỌN ĐẶC TRƯNG
# ===========================================================================
df = pd.read_pickle(r"project/rossmann_clean.pkl")

NUM_FEATURES = [
    "CompetitionDistance", "comp_open_months", "comp_unknown",
    "Promo", "Promo2", "promo2_weeks", "SchoolHoliday",
    "DayOfWeek", "Month", "Day", "WeekOfYear",
    "lag_42", "roll_mean_28", "roll_mean_91",
]
CAT_FEATURES = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"]
FEATURES = NUM_FEATURES + CAT_FEATURES

# ===========================================================================
# B2. CHIA TRAIN / TEST THEO THỜI GIAN  (KHÔNG xáo trộn)
# ===========================================================================
df = df.sort_values(["Date", "Store"]).reset_index(drop=True)
test_start = df["Date"].max() - pd.Timedelta(days=TEST_DAYS - 1)

train = df[df["Date"] < test_start].reset_index(drop=True)
test = df[df["Date"] >= test_start].reset_index(drop=True)

X_train, y_train = train[FEATURES], train["Sales"]
X_test, y_test = test[FEATURES], test["Sales"]

print(f"Train: {train['Date'].min().date()} -> {train['Date'].max().date()}  ({len(train):,} dòng)")
print(f"Test : {test['Date'].min().date()} -> {test['Date'].max().date()}  ({len(test):,} dòng)")

# ===========================================================================
# B3. CÁC FOLD KIỂM ĐỊNH ROLLING-ORIGIN
#     Fold 1: train = mọi ngày trước cửa sổ 1, kiểm định = cửa sổ 1 (42 ngày)
#     Fold 2: train dài thêm, kiểm định = cửa sổ 2 ... luôn "quá khứ -> tương lai"
# ===========================================================================
folds = []
last_train_day = train["Date"].max()
for k in range(N_FOLDS, 0, -1):
    val_end = last_train_day - pd.Timedelta(days=(k - 1) * VAL_DAYS)
    val_start = val_end - pd.Timedelta(days=VAL_DAYS - 1)
    tr_idx = np.where(train["Date"] < val_start)[0]
    va_idx = np.where((train["Date"] >= val_start) & (train["Date"] <= val_end))[0]
    # Lấy mẫu MỘT LẦN cho mỗi fold -> mọi mô hình học trên cùng dữ liệu (so sánh công bằng)
    if len(tr_idx) > FIT_ROWS:
        tr_idx = np.sort(rng.choice(tr_idx, size=FIT_ROWS, replace=False))
    folds.append((tr_idx, va_idx))
    print(f"Fold {len(folds)}: kiểm định {val_start.date()} -> {val_end.date()}")

# ===========================================================================
# B4. CHỈ SỐ ĐÁNH GIÁ
#   MAE   : sai số tuyệt đối trung bình, đơn vị EUR/cửa hàng/ngày
#           -> CHỈ SỐ CHÍNH: dễ đổi ra tiền, ra giờ công cho quyết định xếp ca
#   RMSPE : sai số theo %, chỉ số chính thức của cuộc thi Kaggle
#           -> so sánh công bằng giữa cửa hàng lớn và nhỏ
#   Bias  : trung bình (dự báo - thực tế); > 0 = dự báo thừa, < 0 = dự báo thiếu
#           -> thiếu người/hết hàng và thừa lương/tồn kho có chi phí khác nhau
# ===========================================================================
def mae(y_true, y_pred):
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))


def rmspe(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))


def bias(y_true, y_pred):
    return np.mean(np.asarray(y_pred) - np.asarray(y_true))


def all_scores(y_true, y_pred):
    return {"MAE": mae(y_true, y_pred), "RMSPE": rmspe(y_true, y_pred), "Bias": bias(y_true, y_pred)}


# Dạng "scorer" để GridSearchCV dùng (sklearn quy ước: điểm càng LỚN càng tốt,
# nên MAE và RMSPE được đổi dấu âm; Bias giữ nguyên)
SCORERS = {
    "MAE": make_scorer(mae, greater_is_better=False),
    "RMSPE": make_scorer(rmspe, greater_is_better=False),
    "Bias": make_scorer(bias),
}

# ===========================================================================
# B5. PIPELINE TÁI LẬP ĐƯỢC
#   - Số: điền khuyết bằng median (+ cột cờ "đã bị thiếu") -> chuẩn hóa
#   - Chữ: điền "Missing" -> one-hot
#   - Mô hình học trên log(1+Sales) vì Sales lệch phải; dự báo được đổi ngược về EUR
#   Median / mean / std chỉ được học từ phần TRAIN của mỗi lần fit -> không rò rỉ.
# ===========================================================================
def make_pipeline(model):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("num", numeric, NUM_FEATURES),
        ("cat", categorical, CAT_FEATURES),
    ])
    pipe = Pipeline([("prep", prep), ("model", model)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


# ===========================================================================
# B6. BASELINE: "doanh số cùng thứ, 6 tuần trước"
#     (thiếu thì dùng trung bình 28 ngày; vẫn thiếu thì dùng median của train)
# ===========================================================================
train_median = y_train.median()


def baseline_predict(X):
    return X["lag_42"].fillna(X["roll_mean_28"]).fillna(train_median).to_numpy()


baseline_folds = [all_scores(y_train.iloc[va], baseline_predict(X_train.iloc[va]))
                  for _, va in folds]
baseline_folds = pd.DataFrame(baseline_folds)
print("\nBaseline trên các fold:\n", baseline_folds.round(3))

# ===========================================================================
# B7. TINH CHỈNH SIÊU THAM SỐ + KIỂM TRA OVERFITTING
#   Mỗi họ mô hình có 1 lưới tham số nhỏ. GridSearchCV thử từng tổ hợp trên
#   các fold rolling-origin ở B3, tính sai số trên train VÀ trên kiểm định.
#   Overfitting = sai số train rất thấp nhưng kiểm định cao (khoảng cách lớn).
#   Tham số chống overfitting:
#     Ridge         : alpha (phạt hệ số lớn)
#     Random Forest : max_depth (độ sâu cây), min_samples_leaf (số mẫu tối thiểu ở lá)
#     MLP (ANN)     : alpha (phạt trọng số lớn), kích thước mạng, early_stopping
# ===========================================================================
P = "regressor__model__"     # tiền tố để GridSearchCV tìm tới mô hình bên trong pipeline

MODELS = {
    "Ridge (hồi quy tuyến tính)": (
        Ridge(),
        {P + "alpha": [0.1, 1, 10, 100]},
    ),
    "Random Forest (cây quyết định)": (
        RandomForestRegressor(n_estimators=40, max_features=0.5, n_jobs=-1, random_state=SEED),
        {P + "max_depth": [4, 8, 12, 16, 20], P + "min_samples_leaf": [5, 50]},
    ),
    "MLP (mạng nơ-ron)": (
        MLPRegressor(early_stopping=True, max_iter=200, random_state=SEED),
        {P + "hidden_layer_sizes": [(32,), (64, 32)], P + "alpha": [0.0001, 0.01, 1.0]},
    ),
}

best_params = {}      # tham số tốt nhất của từng họ
cv_rows = []          # bảng so sánh
fold_mae = {"Baseline": baseline_folds["MAE"].to_numpy()}   # để vẽ boxplot

for name, (model, grid) in MODELS.items():
    print(f"\nĐang tinh chỉnh: {name} ...")
    search = GridSearchCV(
        make_pipeline(model), grid, cv=folds,
        scoring=SCORERS, refit=False, return_train_score=True, n_jobs=1,
    )
    search.fit(X_train, y_train)
    res = pd.DataFrame(search.cv_results_)

    # Bảng đầy đủ mọi tổ hợp tham số: train MAE vs kiểm định MAE
    table = pd.DataFrame({
        "params": res["params"].astype(str),
        "train MAE": -res["mean_train_MAE"],
        "val MAE": -res["mean_test_MAE"],
        "val MAE SD": res["std_test_MAE"],
    })
    table["khoảng cách (overfit)"] = table["val MAE"] - table["train MAE"]
    print(table.round(1).to_string(index=False))
    table.round(2).to_csv(f"results/tuning_{name.split()[0]}.csv", index=False)

    # Chọn tổ hợp có MAE kiểm định thấp nhất
    i = (-res["mean_test_MAE"]).idxmin()
    best_params[name] = res.loc[i, "params"]
    fold_mae[name] = np.array([-res.loc[i, f"split{f}_test_MAE"] for f in range(N_FOLDS)])
    cv_rows.append({
        "Mô hình": name,
        "Tham số tốt nhất": {k.replace(P, ""): v for k, v in best_params[name].items()},
        "MAE": f"{-res.loc[i, 'mean_test_MAE']:.0f} ± {res.loc[i, 'std_test_MAE']:.0f}",
        "RMSPE": f"{-res.loc[i, 'mean_test_RMSPE']:.3f} ± {res.loc[i, 'std_test_RMSPE']:.3f}",
        "Bias": f"{res.loc[i, 'mean_test_Bias']:.0f}",
        "_mae": -res.loc[i, "mean_test_MAE"],
    })

    # Hình overfitting cho Random Forest: sai số theo độ sâu cây
    if name.startswith("Random Forest"):
        sub = res[res["param_" + P + "min_samples_leaf"] == 50]
        depth = sub["param_" + P + "max_depth"].astype(int)
        plt.figure(figsize=(6, 4))
        plt.plot(depth, -sub["mean_train_MAE"], "o-", label="Train")
        plt.errorbar(depth, -sub["mean_test_MAE"], yerr=sub["std_test_MAE"],
                     fmt="o-", capsize=4, label="Kiểm định (± SD giữa các fold)")
        plt.xlabel("max_depth (độ sâu tối đa của cây)")
        plt.ylabel("MAE (EUR / cửa hàng / ngày)")
        plt.title("Random Forest: sai số theo độ sâu (min_samples_leaf = 50)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/overfitting_random_forest.png", dpi=150)
        plt.close()

# Bảng so sánh: baseline vs 3 họ mô hình (trên kiểm định, KHÔNG phải test)
cv_rows.insert(0, {
    "Mô hình": "Baseline (cùng thứ, 6 tuần trước)", "Tham số tốt nhất": "-",
    "MAE": f"{baseline_folds['MAE'].mean():.0f} ± {baseline_folds['MAE'].std(ddof=0):.0f}",
    "RMSPE": f"{baseline_folds['RMSPE'].mean():.3f} ± {baseline_folds['RMSPE'].std(ddof=0):.3f}",
    "Bias": f"{baseline_folds['Bias'].mean():.0f}",
    "_mae": baseline_folds["MAE"].mean(),
})
cv_table = pd.DataFrame(cv_rows)
print("\n=== SO SÁNH TRÊN KIỂM ĐỊNH ROLLING-ORIGIN (trung bình ± SD giữa các fold) ===")
print(cv_table.drop(columns="_mae").to_string(index=False))
cv_table.drop(columns="_mae").to_csv("results/cv_comparison.csv", index=False)

# CHỌN mô hình tốt nhất theo MAE KIỂM ĐỊNH (không nhìn tập test)
best_name = cv_table.iloc[1:].sort_values("_mae").iloc[0]["Mô hình"]
print("\n>>> Mô hình được chọn (theo kiểm định):", best_name)

# Hình: MAE từng fold của từng mô hình
plt.figure(figsize=(7, 4))
plt.boxplot(list(fold_mae.values()))
plt.xticks(range(1, len(fold_mae) + 1), [n.split(" (")[0] for n in fold_mae], rotation=15)
plt.ylabel("MAE kiểm định (EUR / cửa hàng / ngày)")
plt.title(f"Sai số trên {N_FOLDS} fold rolling-origin")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/cv_boxplot.png", dpi=150)
plt.close()

# ===========================================================================
# B8. ĐÁNH GIÁ TRÊN TẬP TEST (CHỈ MỞ 1 LẦN)
#   Huấn luyện lại mỗi mô hình (với tham số tốt nhất) trên tập train,
#   rồi dự báo 42 ngày test. Mô hình "được chọn" đã quyết định ở B7.
# ===========================================================================
fit_idx = np.sort(rng.choice(len(X_train), size=min(FIT_ROWS, len(X_train)), replace=False))

test_rows = {"Baseline (cùng thứ, 6 tuần trước)": all_scores(y_test, baseline_predict(X_test))}
final_models = {}
for name, (model, _) in MODELS.items():
    m = make_pipeline(model)
    m.set_params(**best_params[name])
    m.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
    final_models[name] = m
    test_rows[name] = all_scores(y_test, np.clip(m.predict(X_test), 0, None))

test_table = pd.DataFrame(test_rows).T
print("\n=== KẾT QUẢ TRÊN TẬP TEST (42 ngày cuối) ===")
print(test_table.round(3).to_string())
test_table.round(3).to_csv("results/test_results.csv")

best_model = final_models[best_name]
pred_test = np.clip(best_model.predict(X_test), 0, None)
base_mae = test_table.iloc[0]["MAE"]
best_mae = test_table.loc[best_name, "MAE"]
print(f"\n{best_name}: MAE {best_mae:,.0f} EUR so với baseline {base_mae:,.0f} EUR "
      f"-> giảm {(1 - best_mae / base_mae) * 100:.1f}%")

# Hình: thực tế vs dự báo cho 1 cửa hàng trong 42 ngày test
store_id = test["Store"].iloc[0]
s = test["Store"] == store_id
plt.figure(figsize=(8, 4))
plt.plot(test.loc[s, "Date"], y_test[s], "o-", label="Thực tế")
plt.plot(test.loc[s, "Date"], pred_test[s], "s--", label=best_name.split(" (")[0])
plt.plot(test.loc[s, "Date"], baseline_predict(X_test[s]), ":", label="Baseline")
plt.xlabel("Ngày")
plt.ylabel("Doanh số (EUR)")
plt.title(f"Cửa hàng {store_id}: thực tế vs dự báo trên tập test")
plt.legend()
plt.grid(alpha=0.3)
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig("results/test_actual_vs_pred.png", dpi=150)
plt.close()


# ===========================================================================
# B9. DIỄN GIẢI MÔ HÌNH
#   (a) Permutation importance: xáo trộn 1 đặc trưng -> MAE tăng bao nhiêu?
#       Tăng nhiều = mô hình phụ thuộc nhiều vào đặc trưng đó.
#   (b) Sai số theo nhóm: mô hình dự báo kém ở đâu?
# ===========================================================================
sample = rng.choice(len(X_test), size=min(10_000, len(X_test)), replace=False)
imp = permutation_importance(best_model, X_test.iloc[sample], y_test.iloc[sample],
                             scoring="neg_mean_absolute_error", n_repeats=3, random_state=SEED)
imp_table = pd.DataFrame({"Đặc trưng": FEATURES,
                          "MAE tăng khi xáo trộn (EUR)": imp.importances_mean}
                         ).sort_values("MAE tăng khi xáo trộn (EUR)", ascending=False)
print("\n=== ĐỘ QUAN TRỌNG ĐẶC TRƯNG ===")
print(imp_table.round(1).to_string(index=False))
imp_table.round(2).to_csv("results/feature_importance.csv", index=False)

top = imp_table.head(10).iloc[::-1]
plt.figure(figsize=(7, 4.5))
plt.barh(top["Đặc trưng"], top["MAE tăng khi xáo trộn (EUR)"])
plt.xlabel("MAE tăng khi xáo trộn đặc trưng (EUR)")
plt.title(f"Độ quan trọng đặc trưng - {best_name.split(' (')[0]}")
plt.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=150)
plt.close()

# Sai số theo nhóm
test_err = test[["StoreType", "Promo", "DayOfWeek"]].copy()
test_err["abs_err"] = np.abs(y_test - pred_test)
test_err["err"] = pred_test - y_test
for col in ["StoreType", "Promo", "DayOfWeek"]:
    g = test_err.groupby(col).agg(MAE=("abs_err", "mean"), Bias=("err", "mean"), so_dong=("err", "size"))
    print(f"\nSai số theo {col}:\n", g.round(0).to_string())

print("\nXong. Bảng và hình nằm trong thư mục results/")
