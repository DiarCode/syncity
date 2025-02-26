import json
import os
import pickle
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

# Load the enhanced data
data = pd.read_csv("./data/passflow_enhanced.csv")

# Filter out invalid target values
# Keep only positive target values
data = data[data["net_passenger_change"] > 0]

# Define features and target
features = [
    "bus_stop_id", "bus_id", "hour", "day_of_week", "enter_sum", "exit_sum",
    "net_flow", "enter_sum_lag1", "exit_sum_lag1", "tickets_lag1",
    "enter_rolling_mean_3", "exit_rolling_mean_3", "tickets_rolling_mean_3",
    "is_weekend", "is_peak_hour", "rolling_tickets_5min"
]
target = "net_passenger_change"

# Log-transform the target
data[target] = np.log1p(data[target])

# Sort by time for time-based split
data = data.sort_values("bus_board_computer_sent_time")

# Identify unique dates
unique_dates = pd.to_datetime(
    data["bus_board_computer_sent_time"]).dt.date.unique()
if len(unique_dates) < 2:
    raise ValueError("Not enough unique dates to split.")

# Split into training and testing
train_dates = unique_dates[:-1]
test_date = unique_dates[-1]

train_df = data[pd.to_datetime(
    data["bus_board_computer_sent_time"]).dt.date.isin(train_dates)]
test_df = data[pd.to_datetime(
    data["bus_board_computer_sent_time"]).dt.date == test_date]

if test_df.empty:
    split_idx = int(len(data) * 0.8)
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]
    print("Fallback: Using last 20% of data as test set.")

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

if X_test.empty:
    raise ValueError("Test set is empty after fallback.")

# Define MAPE scorer


def mape_scorer(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0:
        return 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


mape_scorer_custom = make_scorer(mape_scorer, greater_is_better=False)

# Hyperparameter space for LightGBM
param_distributions = {
    'num_leaves': [15, 31, 63, 127],
    'max_depth': [3, 5, 7, -1],
    'learning_rate': [0.01, 0.05, 0.1],
    'feature_fraction': [0.6, 0.8, 1.0],
    'bagging_fraction': [0.6, 0.8, 1.0],
    'bagging_freq': [1, 5, 10],
    'reg_alpha': [0, 1, 5, 10],
    'reg_lambda': [0, 1, 5, 10],
    'n_estimators': [100, 200, 300, 500]
}

lgbm_model = lgb.LGBMRegressor(random_state=42)

# Time-series split
tscv = TimeSeriesSplit(n_splits=3)

# Hyperparameter tuning
search = RandomizedSearchCV(
    lgbm_model,
    param_distributions=param_distributions,
    n_iter=30,
    scoring=mape_scorer_custom,
    cv=tscv,
    random_state=42,
    verbose=2,
    n_jobs=-1
)

search.fit(X_train, y_train)

# Best parameters and retraining
best_params = search.best_params_
best_lgb_model = search.best_estimator_
best_lgb_model.fit(X_train, y_train)

# Predictions
y_pred = np.expm1(best_lgb_model.predict(X_test))
y_test_actual = np.expm1(y_test)

# Metrics
mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
mask = y_test_actual != 0
mape = np.mean(np.abs((y_test_actual[mask] - y_pred[mask]) /
               y_test_actual[mask])) * 100 if mask.sum() > 0 else np.nan

# Print results
print("Best parameters found:", best_params)
print(f"Optimized LGBM MAE: {mae:.2f}")
print(f"Optimized LGBM RMSE: {rmse:.2f}")
print(f"Optimized LGBM MAPE: {mape:.2f}%")

# Save model and results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs('./models', exist_ok=True)
model_filename = f'./models/lgbm_{timestamp}.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(best_lgb_model, f)

results = {
    'model': 'LGBMRegressor',
    'timestamp': timestamp,
    'features': features,
    'target': target,
    'train_dates': [str(d) for d in train_dates],
    'test_date': str(test_date),
    'best_params': best_params,
    'MAE': mae,
    'RMSE': rmse,
    'MAPE': mape if not np.isnan(mape) else None
}

os.makedirs('./results', exist_ok=True)
results_filename = f'./results/lgbm_results_{timestamp}.json'
with open(results_filename, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Model saved to {model_filename}")
print(f"Results saved to {results_filename}")

# Log the run
log_filename = './results/training_log.csv'
log_entry = pd.DataFrame([{
    'timestamp': timestamp,
    'model': 'LGBMRegressor',
    'MAE': mae,
    'RMSE': rmse,
    'MAPE': mape if not np.isnan(mape) else None,
    'train_dates': ';'.join([str(d) for d in train_dates]),
    'test_date': str(test_date),
    'best_params': str(best_params)
}])

if not os.path.exists(log_filename):
    log_entry.to_csv(log_filename, index=False)
else:
    log_entry.to_csv(log_filename, mode='a', header=False, index=False)

print(f"Run logged in {log_filename}")
