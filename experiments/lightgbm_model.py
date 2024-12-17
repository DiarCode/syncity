import json
import os
import pickle
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (make_scorer, mean_absolute_error,
                             mean_squared_error)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

# Load prepared dataset and do same steps as before
df = pd.read_csv('./data/passflow_prepared.csv')
df['bus_board_computer_sent_time'] = pd.to_datetime(
    df['bus_board_computer_sent_time'], errors='coerce')
if not np.issubdtype(df['bus_board_computer_sent_time'].dtype, np.datetime64):
    raise ValueError("Column 'bus_board_computer_sent_time' is not datetime.")

target_col = 'enter_sum'
feature_cols = [
    'hour', 'day_of_week', 'net_passenger_change',
    'enter_sum_lag1', 'exit_sum_lag1', 'tickets_lag1',
    'enter_rolling_mean_3', 'exit_rolling_mean_3', 'tickets_rolling_mean_3',
    'route_number', 'bus_stop_id', 'bus_id'
]

df = df.sort_values('bus_board_computer_sent_time')
unique_dates = df['bus_board_computer_sent_time'].dt.date.unique()
if len(unique_dates) < 2:
    raise ValueError(f"Not enough unique dates. Found: {unique_dates}")

train_dates = unique_dates[:-1]
test_date = unique_dates[-1]

train_df = df[df['bus_board_computer_sent_time'].dt.date.isin(train_dates)]
test_df = df[df['bus_board_computer_sent_time'].dt.date == test_date]

if test_df.empty:
    split_idx = int(len(df)*0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    print("No test data found for the last date. Using last 20% of data as test set.")

X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]
y_test = test_df[target_col]
if X_test.empty:
    raise ValueError("Test set is empty after fallback.")


def mape_scorer(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0:
        return 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


mape_scorer_custom = make_scorer(mape_scorer, greater_is_better=False)
tscv = TimeSeriesSplit(n_splits=3)

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

print("Best parameters found:", search.best_params_)
print("Best CV score (MAPE):", -search.best_score_)

# Retrain best model
best_lgb_model = search.best_estimator_
best_lgb_model.fit(X_train, y_train)

y_pred = best_lgb_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mask = y_test != 0
if mask.sum() == 0:
    mape = np.nan
else:
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100

print("Optimized LGBM MAE:", mae)
print("Optimized LGBM RMSE:", rmse)
if not np.isnan(mape):
    print("Optimized LGBM MAPE: {:.2f}%".format(mape))

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs('./models', exist_ok=True)
model_filename = f'./models/lgbm_{timestamp}.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(best_lgb_model, f)

results = {
    'model': 'LGBMRegressor',
    'timestamp': timestamp,
    'features': feature_cols,
    'target': target_col,
    'train_dates': [str(d) for d in train_dates],
    'test_date': str(test_date),
    'best_params': search.best_params_,
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

log_filename = './results/training_log.csv'
log_entry = pd.DataFrame([{
    'timestamp': timestamp,
    'model': 'LGBMRegressor',
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
    log_entry.to_csv(log_filename, mode='a', header=False, index=False)

print(f"Run logged in {log_filename}")
