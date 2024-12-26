import json
import os
from datetime import datetime

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Parameters
DATA_PATH = './data/passflow_prepared.csv'
DATE_COL = 'bus_board_computer_sent_time'
TARGET_COL = 'enter_sum'
FEATURE_COLS = [
    'hour', 'day_of_week', 'net_passenger_change',
    'enter_sum_lag1', 'exit_sum_lag1', 'tickets_lag1',
    'enter_rolling_mean_3', 'exit_rolling_mean_3', 'tickets_rolling_mean_3',
    'route_number', 'bus_stop_id', 'bus_id'
]
WINDOW_SIZE = 24  # use last 24 points (e.g., last 24 hours if hourly)
EPOCHS = 50

# Load data
df = pd.read_csv(DATA_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
df = df.dropna(subset=[DATE_COL])
df = df.sort_values(DATE_COL)

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

# Log transform the target
y = np.log1p(y)

# Time-based split
unique_dates = df[DATE_COL].dt.date.unique()
if len(unique_dates) < 2:
    raise ValueError("Not enough unique dates.")

train_dates = unique_dates[:-1]
test_date = unique_dates[-1]

train_mask = df[DATE_COL].dt.date.isin(train_dates)
test_mask = df[DATE_COL].dt.date == test_date

if test_mask.sum() == 0:
    # fallback: last 20% as test
    split_idx = int(len(X)*0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    print("No test data found for the last date. Using last 20% of data as test set.")
else:
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def create_sequences(X, y, window_size):
    Xs, ys = [], []
    for i in range(len(X)-window_size):
        Xs.append(X[i:(i+window_size), :])
        ys.append(y[i+window_size])
    return np.array(Xs), np.array(ys)


X_train_seq, y_train_seq = create_sequences(X_train, y_train, WINDOW_SIZE)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, WINDOW_SIZE)

if len(X_test_seq) == 0:
    raise ValueError("Not enough test data after sequencing.")

# Hypermodel for LSTM with tuner


def build_model(hp):
    model = keras.Sequential()
    hp_units = hp.Int('units', min_value=64, max_value=256, step=64)
    model.add(layers.LSTM(hp_units, return_sequences=True,
              input_shape=(WINDOW_SIZE, X_train_seq.shape[-1])))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(hp_units // 2))
    model.add(layers.Dense(1))
    hp_learning_rate = hp.Choice('learning_rate', [1e-3, 5e-4, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='mse')
    return model


tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=1,
    directory='lstm_tuner_dir',
    project_name='lstm_hypersearch',
    overwrite=True
)

tuner.search(X_train_seq, y_train_seq,
             validation_split=0.2, epochs=10, verbose=1)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best Hyperparameters found:", best_hps.values)

# Build final model
model = tuner.hypermodel.build(best_hps)

early_stop = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

history = model.fit(X_train_seq, y_train_seq,
                    validation_split=0.2,
                    epochs=EPOCHS,
                    callbacks=[early_stop, reduce_lr],
                    verbose=1)

# Predict
y_pred_seq = model.predict(X_test_seq)
# Invert log transform
y_test_inv = np.expm1(y_test_seq)
y_pred_inv = np.expm1(y_pred_seq.ravel())

mae = mean_absolute_error(y_test_inv, y_pred_inv)
mse = mean_squared_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mse)
mask = y_test_inv != 0
mape = np.mean(np.abs((y_test_inv[mask] - y_pred_inv[mask]) /
               y_test_inv[mask])) * 100 if mask.sum() > 0 else np.nan

print("LSTM MAE:", mae)
print("LSTM RMSE:", rmse)
if not np.isnan(mape):
    print(f"LSTM MAPE: {mape:.2f}%")
else:
    print("MAPE not available due to zero values in target.")

# Save model and results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs('./models', exist_ok=True)
model_filename = f'./models/lstm_{timestamp}.h5'
model.save(model_filename)

results = {
    'model': 'LSTM',
    'timestamp': timestamp,
    'window_size': WINDOW_SIZE,
    'best_params': best_hps.values,
    'MAE': mae,
    'RMSE': rmse,
    'MAPE': mape if not np.isnan(mape) else None
}

os.makedirs('./results', exist_ok=True)
results_filename = f'./results/lstm_results_{timestamp}.json'
with open(results_filename, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Model saved to {model_filename}")
print(f"Results saved to {results_filename}")

log_filename = './results/training_log.csv'
log_entry = pd.DataFrame([{
    'timestamp': timestamp,
    'model': 'LSTM',
    'window_size': WINDOW_SIZE,
    'best_params': str(best_hps.values),
    'MAE': mae,
    'RMSE': rmse,
    'MAPE': mape if not np.isnan(mape) else None
}])

if not os.path.exists(log_filename):
    log_entry.to_csv(log_filename, index=False)
else:
    log_entry.to_csv(log_filename, mode='a', header=False, index=False)

print(f"Run logged in {log_filename}")
print(f"Run logged in {log_filename}")
