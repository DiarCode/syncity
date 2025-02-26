import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import OrdinalEncoder

# Load and preprocess the data
print("Loading and preprocessing data...")
data = pd.read_csv("./data/passflow_enhanced.csv")
data["bus_board_computer_sent_time"] = pd.to_datetime(
    data["bus_board_computer_sent_time"], errors="coerce"
)
data = data.dropna(subset=["bus_board_computer_sent_time"])
data = data.sort_values("bus_board_computer_sent_time")
data = data[data["net_passenger_change"] > 0]  # Filter positive targets
print(f"Data shape after preprocessing: {data.shape}")

# Define features and target
features = [
    "bus_stop_id", "bus_id", "hour", "day_of_week", "enter_sum", "exit_sum",
    "net_flow", "enter_sum_lag1", "exit_sum_lag1", "tickets_lag1",
    "enter_rolling_mean_3", "exit_rolling_mean_3", "tickets_rolling_mean_3",
    "is_weekend", "is_peak_hour", "rolling_tickets_5min"
]
target = "net_passenger_change"

# Encode categorical features
print("Encoding categorical features...")
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
data[["bus_stop_id", "bus_id"]] = encoder.fit_transform(
    data[["bus_stop_id", "bus_id"]])
print("Categorical features encoded.")

# Prepare features and target
X = data[features]
y = np.log1p(data[target])  # Log-transform target for stability
print(f"Features shape: {X.shape}, Target shape: {y.shape}")

# Time-based train-test split (preserve temporal order)
print("Performing time-based train-test split...")
test_size = 0.2
split_idx = int(len(X) * (1 - test_size))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Hyperparameter tuning with regularization
print("Setting up hyperparameter grid for tuning...")
param_grid = {
    "n_estimators": [200, 300],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 4],
    "min_samples_split": [20, 30],
    "min_samples_leaf": [10, 15],
    "subsample": [0.7, 0.8],
    "max_features": ['sqrt', 0.5],
    "validation_fraction": [0.1],  # Early stopping
    "n_iter_no_change": [10]       # Early stopping
}

# Time-series cross-validation
print("Setting up time-series cross-validation...")
tscv = TimeSeriesSplit(n_splits=5)

# Grid search with cross-validation
print("Starting grid search with cross-validation...")
grid_search = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train, y_train)
print("Grid search completed.")

# Get best model with early stopping
best_gb_model = grid_search.best_estimator_
print(f"Best model parameters: {grid_search.best_params_}")

# Feature importance analysis
print("Analyzing feature importances...")
feature_importances = best_gb_model.feature_importances_
important_features = [features[i]
                      for i in np.where(feature_importances > 0.01)[0]]
print(f"Important features: {important_features}")

# Retrain the model using only important features
print("Retraining model with important features only...")
X_train_filtered = X_train[important_features]
X_test_filtered = X_test[important_features]

# Retrain the best model on the filtered features
best_gb_model.fit(X_train_filtered, y_train)

# Predict on the test set using the filtered features
print("Predicting on test set with filtered features...")
y_pred = np.expm1(best_gb_model.predict(
    X_test_filtered))  # Reverse log-transform
y_test_actual = np.expm1(y_test)

# Evaluate model performance
print("Evaluating model performance...")
epsilon = 1e-8
mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
mape = np.mean(np.abs((y_test_actual - y_pred) /
               np.maximum(np.abs(y_test_actual), epsilon))) * 100
smape = 100 * np.mean(2 * np.abs(y_test_actual - y_pred) /
                      (np.abs(y_test_actual) + np.abs(y_pred)))
r2 = r2_score(y_test_actual, y_pred)

# Print results
print("\nModel Performance Metrics (with important features only):")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"SMAPE: {smape:.2f}%")
print(f"R^2 Score: {r2:.2f}")

# Save results
print("Saving results...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("./results", exist_ok=True)
results_filename = f"./results/gradient_boosting_results_{timestamp}.json"
results = {
    'model': 'GradientBoostingRegressor',
    'timestamp': timestamp,
    'features': features,
    'important_features': important_features,
    'target': target,
    'MAE': mae,
    'RMSE': rmse,
    'MAPE': mape,
    'SMAPE': smape,
    'R2': r2,
    'best_params': grid_search.best_params_
}
with open(results_filename, "w") as f:
    json.dump(results, f, indent=4)
print(f"Results saved to {results_filename}")

# Visualize predictions vs actuals
print("Plotting predictions vs actuals...")
plt.figure(figsize=(10, 6))
plt.scatter(y_test_actual, y_pred, alpha=0.7)
plt.plot([0, max(y_test_actual)], [0, max(y_test_actual)],
         color="red", linestyle="--", linewidth=2)
plt.title("Gradient Boosting Regressor: Predictions vs Actuals")
plt.xlabel("Actual Net Passenger Change")
plt.ylabel("Predicted Net Passenger Change")
plt.xlim([0, max(y_test_actual)])
plt.ylim([0, max(y_pred)])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Visualize residuals
print("Plotting residual distribution...")
plt.figure(figsize=(10, 6))
residuals = y_test_actual - y_pred
plt.hist(residuals, bins=50, edgecolor="k", alpha=0.7)
plt.title("Residual Distribution (Gradient Boosting Regressor)")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("Process completed successfully!")
