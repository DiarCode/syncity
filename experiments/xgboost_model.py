import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

# Load the enhanced data
data = pd.read_csv("./data/passflow_enhanced.csv")
data["bus_board_computer_sent_time"] = pd.to_datetime(
    data["bus_board_computer_sent_time"], errors="coerce")
data = data.dropna(subset=["bus_board_computer_sent_time"])
data = data.sort_values("bus_board_computer_sent_time")
data = data[data["net_passenger_change"] > 0]  # Filter positive targets

# Define features and target
features = [
    "bus_stop_id", "bus_id", "hour", "day_of_week", "enter_sum", "exit_sum",
    "net_flow", "enter_sum_lag1", "exit_sum_lag1", "tickets_lag1",
    "enter_rolling_mean_3", "exit_rolling_mean_3", "tickets_rolling_mean_3",
    "is_weekend", "is_peak_hour", "rolling_tickets_5min"
]
target = "net_passenger_change"

X = data[features]
y = np.log1p(data[target])  # Log-transform target

# Time-based split
unique_dates = data["bus_board_computer_sent_time"].dt.date.unique()
if len(unique_dates) < 2:
    raise ValueError("Not enough unique dates.")

train_dates = unique_dates[:-1]
test_date = unique_dates[-1]

train_mask = data["bus_board_computer_sent_time"].dt.date.isin(train_dates)
test_mask = data["bus_board_computer_sent_time"].dt.date == test_date

if test_mask.sum() == 0:
    # Fallback: use last 20% as test
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
    print("No test data found for the last date. Using last 20% of data as test set.")
else:
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

# Define MAPE scorer


def mape_scorer(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0:
        return 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


mape_scorer_custom = make_scorer(mape_scorer, greater_is_better=False)

# Parameter search space for XGBoost
param_distributions = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 1, 5, 10],
    'reg_lambda': [0, 1, 5, 10]
}

tscv = TimeSeriesSplit(n_splits=3)

xgb_model = XGBRegressor(random_state=42, n_jobs=-1)

search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_distributions,
    n_iter=30,
    scoring=mape_scorer_custom,
    cv=tscv,
    random_state=42,
    verbose=2,
    n_jobs=-1
)

search.fit(X_train, y_train)

print("Best parameters found:", search.best_params_)
print("Best CV score (MAPE):", -search.best_score_)

# Retrain the model on full training data with best params
best_xgb_model = search.best_estimator_
best_xgb_model.fit(X_train, y_train)

# Predictions
y_pred = best_xgb_model.predict(X_test)
y_test_inv = np.expm1(y_test)  # Invert log transform
y_pred_inv = np.expm1(y_pred)

# Evaluate
mae = mean_absolute_error(y_test_inv, y_pred_inv)
mse = mean_squared_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mse)
mask = y_test_inv != 0
mape = np.mean(np.abs((y_test_inv[mask] - y_pred_inv[mask]) /
               y_test_inv[mask])) * 100 if mask.sum() > 0 else np.nan

print("Optimized XGB MAE:", mae)
print("Optimized XGB RMSE:", rmse)
if not np.isnan(mape):
    print(f"Optimized XGB MAPE: {mape:.2f}%")
else:
    print("MAPE not available due to zero values in target.")

# Save model and results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("./models", exist_ok=True)
model_filename = f"./models/xgb_{timestamp}.pkl"
with open(model_filename, 'wb') as f:
    pickle.dump(best_xgb_model, f)

results = {
    'model': 'XGBRegressor',
    'timestamp': timestamp,
    'features': features,
    'target': target,
    'train_dates': [str(d) for d in train_dates],
    'test_date': str(test_date),
    'best_params': search.best_params_,
    'MAE': mae,
    'RMSE': rmse,
    'MAPE': mape if not np.isnan(mape) else None
}

os.makedirs("./results", exist_ok=True)
results_filename = f"./results/xgb_results_{timestamp}.json"
with open(results_filename, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Model saved to {model_filename}")
print(f"Results saved to {results_filename}")

# Log the run
log_filename = "./results/training_log.csv"
log_entry = pd.DataFrame([{
    'timestamp': timestamp,
    'model': 'XGBRegressor',
    'MAE': mae,
    'RMSE': rmse,
    'MAPE': mape if not np.isnan(mape) else None,
    'train_dates': ';'.join([str(d) for d in train_dates]),
    'test_date': str(test_date),
    'best_params': str(search.best_params_)
}])

if not os.path.exists(log_filename):
    log_entry.to_csv(log_filename, index=False)
else:
    log_entry.to_csv(log_filename, mode="a", header=False, index=False)

print(f"Run logged in {log_filename}")
