import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess the data
data = pd.read_csv("./data/passflow_enhanced.csv")
data["bus_board_computer_sent_time"] = pd.to_datetime(
    data["bus_board_computer_sent_time"], errors="coerce"
)
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
y = np.log1p(data[target])  # Log-transform target for stability

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = np.expm1(lr_model.predict(X_test_scaled))  # Reverse log-transform
y_test_actual = np.expm1(y_test)

# Evaluate model performance
epsilon = 1e-8  # Small constant to avoid division by zero
mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
mape = np.mean(np.abs((y_test_actual - y_pred) /
               np.maximum(np.abs(y_test_actual), epsilon))) * 100
smape = 100 * np.mean(2 * np.abs(y_test_actual - y_pred) /
                      (np.abs(y_test_actual) + np.abs(y_pred)))
r2 = r2_score(y_test_actual, y_pred)

# Print results
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"SMAPE: {smape:.2f}%")
print(f"R^2 Score: {r2:.2f}")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("./results", exist_ok=True)
results_filename = f"./results/linear_regression_results_{timestamp}.json"
results = {
    'model': 'LinearRegression',
    'timestamp': timestamp,
    'features': features,
    'target': target,
    'MAE': mae,
    'RMSE': rmse,
    'MAPE': mape,
    'SMAPE': smape,
    'R2': r2
}
with open(results_filename, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {results_filename}")

# Visualize predictions vs actuals
plt.figure(figsize=(10, 6))
plt.scatter(y_test_actual, y_pred, alpha=0.7)
plt.plot([0, max(y_test_actual)], [0, max(y_test_actual)],
         color="red", linestyle="--", linewidth=2)
plt.title("Linear Regression: Predictions vs Actuals")
plt.xlabel("Actual Net Passenger Change")
plt.ylabel("Predicted Net Passenger Change")
plt.xlim([0, max(y_test_actual)])
plt.ylim([0, max(y_pred)])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Visualize residuals
plt.figure(figsize=(10, 6))
residuals = y_test_actual - y_pred
plt.hist(residuals, bins=50, edgecolor="k", alpha=0.7)
plt.title("Residual Distribution (Linear Regression)")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
