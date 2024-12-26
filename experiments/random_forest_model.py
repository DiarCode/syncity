import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (make_scorer, mean_absolute_error,
                             mean_squared_error)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

# Load prepared dataset
df = pd.read_csv('./data/passflow_prepared.csv')

# Convert to datetime
df['bus_board_computer_sent_time'] = pd.to_datetime(
    df['bus_board_computer_sent_time'], errors='coerce')

# Verify datetime
if not np.issubdtype(df['bus_board_computer_sent_time'].dtype, np.datetime64):
    raise ValueError(
        "Column 'bus_board_computer_sent_time' is not datetime after conversion.")

# Define features & target
target_col = 'enter_sum'
feature_cols = [
    'hour', 'day_of_week', 'net_passenger_change',
    'enter_sum_lag1', 'exit_sum_lag1', 'tickets_lag1',
    'enter_rolling_mean_3', 'exit_rolling_mean_3', 'tickets_rolling_mean_3',
    'route_number', 'bus_stop_id', 'bus_id'
]

# Sort by time
df = df.sort_values('bus_board_computer_sent_time')

# Identify unique dates
unique_dates = df['bus_board_computer_sent_time'].dt.date.unique()
print("Unique Dates in Data:", unique_dates)

if len(unique_dates) < 2:
    raise ValueError(
        f"Not enough unique dates to perform time-based train/test split. Found: {unique_dates}")

# Use all but the last date as training, last date as test
train_dates = unique_dates[:-1]
test_date = unique_dates[-1]

train_df = df[df['bus_board_computer_sent_time'].dt.date.isin(train_dates)]
test_df = df[df['bus_board_computer_sent_time'].dt.date == test_date]

# If test set is empty, fallback to using the last portion of data as test set
if test_df.empty:
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    print("No test data found for the last date. Using last 20% of data as test set.")

X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]

if X_test.empty:
    raise ValueError(
        "Test set is empty after fallback. Please revise data splitting logic.")

# Use the best parameters found previously
best_params = {
    'max_depth': 3,
    'max_features': None,
    'min_samples_leaf': 3,
    'min_samples_split': 18,
    'n_estimators': 491
}

# Train model with optimized hyperparameters
model = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    max_features=best_params['max_features'],
    min_samples_leaf=best_params['min_samples_leaf'],
    min_samples_split=best_params['min_samples_split'],
    random_state=42
)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate - MAE, RMSE
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Calculate MAPE
mask = y_test != 0
if mask.sum() == 0:
    print("No non-zero values in y_test. Cannot compute MAPE.")
    mape = np.nan
else:
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100

print("Test MAE:", mae)
print("Test RMSE:", rmse)
if not np.isnan(mape):
    print("Test MAPE: {:.2f}%".format(mape))
else:
    print("MAPE could not be calculated due to zero values in the target.")

# Save model and results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs('./models', exist_ok=True)
model_filename = f'./models/random_forest_{timestamp}.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)

results = {
    'model': 'RandomForestRegressor',
    'timestamp': timestamp,
    'features': feature_cols,
    'target': target_col,
    'train_dates': [str(d) for d in train_dates],
    'test_date': str(test_date),
    'best_params': best_params,
    'MAE': mae,
    'RMSE': rmse,
    'MAPE': mape if not np.isnan(mape) else None
}

os.makedirs('./results', exist_ok=True)
results_filename = f'./results/rf_results_{timestamp}.json'
with open(results_filename, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Model saved to {model_filename}")
print(f"Results saved to {results_filename}")

# Log run
log_filename = './results/training_log.csv'
log_entry = pd.DataFrame([{
    'timestamp': timestamp,
    'model': 'RandomForestRegressor',
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
