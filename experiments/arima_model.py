import json
import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# Parameters
DATA_PATH = "./data/passflow_enhanced.csv"
DATE_COL = "bus_board_computer_sent_time"
TARGET_COL = "enter_sum"
RESAMPLE_FREQ = "H"  # Hourly resample
SEASONAL_PERIOD = 24  # Daily seasonality for hourly data

# Load data
df = pd.read_csv(DATA_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[DATE_COL])
df = df.sort_values(DATE_COL)

# Resample and aggregate data
df_agg = df.set_index(DATE_COL).resample(
    RESAMPLE_FREQ)[TARGET_COL].sum().fillna(0)

# Log transformation
df_agg_log = np.log1p(df_agg)

# Time-based split
unique_dates = df_agg_log.index.date
train_dates = unique_dates[:-1]
test_date = unique_dates[-1]

train_series = df_agg_log[df_agg_log.index.date < test_date]
test_series = df_agg_log[df_agg_log.index.date == test_date]

# Fallback for insufficient test data
if test_series.empty:
    split_idx = int(len(df_agg_log) * 0.8)
    train_series = df_agg_log.iloc[:split_idx]
    test_series = df_agg_log.iloc[split_idx:]
    print("Fallback: Using last 20% of data as test set.")

y_train = train_series.values
y_test = test_series.values

# Fit ARIMA model
arima_model = auto_arima(
    y=y_train,
    start_p=1, start_q=1,
    max_p=5, max_q=5,
    seasonal=True,
    m=SEASONAL_PERIOD,
    start_P=0, start_Q=0,
    max_P=2, max_Q=2,
    d=None, D=None,
    stepwise=True,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    random_state=42
)

print("Best ARIMA model:", arima_model.order,
      "seasonal_order:", arima_model.seasonal_order)

# Forecasting
n_periods = len(y_test)
forecast_log = arima_model.predict(n_periods=n_periods)
forecast = np.expm1(forecast_log)
y_test_inv = np.expm1(y_test)

# Evaluate performance
mae = mean_absolute_error(y_test_inv, forecast)
mse = mean_squared_error(y_test_inv, forecast)
rmse = np.sqrt(mse)
mask = y_test_inv != 0
mape = np.mean(np.abs((y_test_inv[mask] - forecast[mask]) /
               y_test_inv[mask])) * 100 if mask.sum() > 0 else np.nan
smape = 100 * np.mean(2 * np.abs(y_test_inv - forecast) /
                      (np.abs(y_test_inv) + np.abs(forecast)))

print(f"ARIMA MAE: {mae:.2f}")
print(f"ARIMA RMSE: {rmse:.2f}")
print(f"ARIMA MAPE: {mape:.2f}%")
print(f"ARIMA SMAPE: {smape:.2f}%")

# Save the model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("./models", exist_ok=True)
model_filename = f"./models/arima_{timestamp}.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(arima_model, f)

# Save results
results = {
    "model": "ARIMA",
    "timestamp": timestamp,
    "order": arima_model.order,
    "seasonal_order": arima_model.seasonal_order,
    "resample_freq": RESAMPLE_FREQ,
    "MAE": mae,
    "RMSE": rmse,
    "MAPE": mape,
    "SMAPE": smape,
}

os.makedirs("./results", exist_ok=True)
results_filename = f"./results/arima_results_{timestamp}.json"
with open(results_filename, "w") as f:
    json.dump(results, f, indent=4)

print(f"Model saved to {model_filename}")
print(f"Results saved to {results_filename}")

# Log the run
log_filename = "./results/training_log.csv"
log_entry = pd.DataFrame([{
    "timestamp": timestamp,
    "model": "ARIMA",
    "order": str(arima_model.order),
    "seasonal_order": str(arima_model.seasonal_order),
    "resample_freq": RESAMPLE_FREQ,
    "MAE": mae,
    "RMSE": rmse,
    "MAPE": mape,
    "SMAPE": smape,
}])
if not os.path.exists(log_filename):
    log_entry.to_csv(log_filename, index=False)
else:
    log_entry.to_csv(log_filename, mode="a", header=False, index=False)

print(f"Run logged in {log_filename}")
