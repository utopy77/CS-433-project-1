import os
import csv
import numpy as np
from helpers import *
from implementations import *

x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path="dataset/")

with open(os.path.join("dataset", "x_train.csv"), newline="") as f:
    feature_names = np.array(next(csv.reader(f))[1:])  # drop 'Id' column

print(f"Initial raw feature count: {x_train.shape[1]}")

# Features to drop
# CVDINFR4 and CVDCRHD4 directly define the target _MICHD.
target_leakage = {
    "CVDINFR4", "CVDCRHD4", "HAREHAB1", "STREHAB1", "_MICHD"
}

# Administrative, dates, tracking, sample stratification and survey weights
admin_and_weights = {
    "_STATE", "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR",
    "DISPCODE", "SEQNO", "PSU", "QSTVER", "QSTLANG",
    "_STSTR", "STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE",
    "_LLCPWT", "_DUALUSE", "_DUALCOR", "_CLLCPWT"
}

columns_to_drop = target_leakage | admin_and_weights

# Remove blacklist columns
valid_col_mask = np.array([name not in columns_to_drop for name in feature_names])
x_train_sub = x_train[:, valid_col_mask]
x_test_sub = x_test[:, valid_col_mask]
features_sub = feature_names[valid_col_mask]

# Remove columns with more than 80% NaN values
missing_ratios = np.isnan(x_train_sub).mean(axis=0)
keep_nan_mask = missing_ratios <= 0.80

x_train_clean = x_train_sub[:, keep_nan_mask]
x_test_clean = x_test_sub[:, keep_nan_mask]
feature_names_clean = features_sub[keep_nan_mask]

print(f"Features remaining after leakage and NaN filtering: {x_train_clean.shape[1]}")

# Clean Survey Sentinel Values (7, 9, 77, 88, 99)
# In BRFSS:
# - For physical/mental days: '88' means 0 days (None).
# - For 1/2 binary flags: 7 ('Don't know') and 9 ('Refused') are not linear scale values.
zero_day_cols = {"PHYSHLTH", "MENTHLTH", "POORHLTH", "CHILDREN"}
binary_sentinel_cols = {
    "BPHIGH4", "BPMEDS", "BLOODCHO", "TOLDHI2", "CVDSTRK3",
    "ASTHMA3", "CHCSCNCR", "CHCOCNCR", "CHCCOPD1", "HAVARTH3",
    "ADDEPEV2", "CHCKIDNY", "DIABETE3", "EXERANY2", "DIFFWALK",
    "DIFFDRES", "DIFFALON", "SMOKE100", "_RFHYPE5", "_RFCHOL",
    "_LTASTH1", "_RFSMOK3", "_RFBMI5", "_RFBING5", "_RFDRHV5"
}

for i, col in enumerate(feature_names_clean):
    # Convert 88 (meaning zero/none) to 0
    if col in zero_day_cols:
        x_train_clean[x_train_clean[:, i] == 88, i] = 0.0
        x_test_clean[x_test_clean[:, i] == 88, i] = 0.0

    # Convert 7 and 9 (meaning "Don't know / Refused") to NaN
    if col in binary_sentinel_cols:
        x_train_clean[np.isin(x_train_clean[:, i], [7, 9]), i] = np.nan
        x_test_clean[np.isin(x_test_clean[:, i], [7, 9]), i] = np.nan

# Compute medians strictly on train data
medians = np.nanmedian(x_train_clean, axis=0)

# Replace all NaNs in both sets using the train medians
nan_train_idx = np.where(np.isnan(x_train_clean))
x_train_clean[nan_train_idx] = np.take(medians, nan_train_idx[1])

nan_test_idx = np.where(np.isnan(x_test_clean))
x_test_clean[nan_test_idx] = np.take(medians, nan_test_idx[1])

# Removing near zero-variance features
col_vars = np.var(x_train_clean, axis=0)
non_zero_var_mask = col_vars > 1e-4

x_train_clean = x_train_clean[:, non_zero_var_mask]
x_test_clean = x_test_clean[:, non_zero_var_mask]
feature_names_clean = feature_names_clean[non_zero_var_mask]

# Standardization
train_means = np.mean(x_train_clean, axis=0)
train_stds = np.std(x_train_clean, axis=0)
train_stds[train_stds == 0] = 1.0  # safety against divide-by-zero

# Normalize using train parameters
tx_train_scaled = (x_train_clean - train_means) / train_stds
tx_test_scaled = (x_test_clean - train_means) / train_stds

# Add intercept
tx_tr = np.c_[np.ones(tx_train_scaled.shape[0]), tx_train_scaled]
tx_te = np.c_[np.ones(tx_test_scaled.shape[0]), tx_test_scaled]

print(f"Final tx_train shape (with intercept): {tx_tr.shape}")
print(f"Final tx_test shape (with intercept):  {tx_te.shape}")

best_lambda, best_loss = cross_validation(
    y_train,
    tx_tr,
    k_fold=5,
    lambdas=np.array([0.0, 1e-4, 1e-3, 1e-2]),
    max_iters=1000,
    gamma=0.01,
)
print("Best lambda:", best_lambda)
print("Best loss:", best_loss)