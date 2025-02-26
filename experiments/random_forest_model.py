import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import randint 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

# Load the enhanced dataset
data = pd.read_csv("./data/passflow_enhanced.csv")

# Convert time columns
data["bus_board_computer_sent_time"] = pd.to_datetime(
    data["bus_board_computer_sent_time"])

# Filter positive target values
data = data[data["net_passenger_change"] > 0]

# Define features and target
target_col = "net_passenger_change"
feature_cols = [
    "hour", "day_of_week", "net_flow", "enter_sum_lag1", "exit_sum_lag1", "tickets_lag1",
    "enter_rolling_mean_3", "exit_rolling_mean_3", "tickets_rolling_mean_3",
    "route_number", "bus_stop_id", "bus_id", "is_weekend", "is_peak_hour", "rolling_tickets_5min"
]

# Log-transform the target
data[target_col] = np.log1p(data[target_col])

# Time-based split
data = data.sort_values("bus_board_computer_sent_time")
unique_dates = data["bus_board_computer_sent_time"].dt.date.unique()

train_dates = unique_dates[:-1]
test_date = unique_dates[-1]

train_df = data[data["bus_board_computer_sent_time"].dt.date.isin(train_dates)]
test_df = data[data["bus_board_computer_sent_time"].dt.date == test_date]

X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]

# Updated Hyperparameter Distribution
param_dist = {
    "n_estimators": randint(100, 300),  # Reduced upper limit
    "max_depth": randint(5, 12),       # Prevent deep trees
    "min_samples_split": randint(5, 20),  # Avoid over-specialization
    "min_samples_leaf": randint(2, 10),   # Increase minimum leaf size
    "max_features": ["sqrt", "log2", None]  # Fixed 'auto' issue
}

# Initialize Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Cross-Validation with RandomizedSearchCV
search = RandomizedSearchCV(
    rf, param_distributions=param_dist, n_iter=50,
    scoring="neg_mean_squared_error", cv=TimeSeriesSplit(n_splits=3),
    random_state=42, n_jobs=-1, verbose=1
)
search.fit(X_train, y_train)

# Best model and parameters
best_model = search.best_estimator_
best_params = search.best_params_

# Train with best parameters
best_model.fit(X_train, y_train)

# Predictions
y_pred = np.expm1(best_model.predict(X_test))
y_test_actual = np.expm1(y_test)

# Metrics calculation
epsilon = 1e-8  # Small constant to avoid division by zero
mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
mape = np.mean(np.abs((y_test_actual - y_pred) /
               np.maximum(np.abs(y_test_actual), epsilon))) * 100
r2 = r2_score(y_test_actual, y_pred)

# Feature Importance
importances = best_model.feature_importances_
feature_importance = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("----- Feature Importance -----")
print(feature_importance)

# Print metrics
print("----- Model Metrics -----")
print(f"Best Parameters: {best_params}")
print(f"Test MAE: {mae:.2f}")
print(f"Test RMSE: {rmse:.2f}")
print(f"Test MAPE: {mape:.2f}%")
print(f"R^2 Score: {r2:.2f}")

# Save model and results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs('./models', exist_ok=True)
model_filename = f'./models/random_forest_{timestamp}.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(best_model, f)

results = {
    "model": "RandomForestRegressor",
    "timestamp": timestamp,
    "features": feature_cols,
    "target": target_col,
    "train_dates": [str(d) for d in train_dates],
    "test_date": str(test_date),
    "best_params": best_params,
    "test_metrics": {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    },
    "feature_importance": feature_importance.to_dict(orient="records")
}

os.makedirs('./results', exist_ok=True)
results_filename = f'./results/rf_results_{timestamp}.json'
with open(results_filename, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Model saved to {model_filename}")
print(f"Results saved to {results_filename}")
