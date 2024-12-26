import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Parameters
DATA_PATH = "./data/passflow_prepared.csv"
DATE_COL = "bus_board_computer_sent_time"
TARGET_COL = "enter_sum"
group_ids = ["bus_stop_id"]
MAX_ENCODER_LENGTH = 12  # Adjusted encoder length
MAX_PREDICTION_LENGTH = 6  # Adjusted prediction length
BATCH_SIZE = 64
EPOCHS = 30
SEED = 42

if __name__ == "__main__":
    # Set seed for reproducibility
    seed_everything(SEED, workers=True)

    # Load and preprocess data
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
    df = df.dropna(subset=[DATE_COL])
    df = df.sort_values(DATE_COL)
    df["bus_stop_id"] = df["bus_stop_id"].astype(str)

    # Generate time index and log-transform the target
    df["time_idx"] = np.arange(len(df))
    df["target_transformed"] = np.log1p(df[TARGET_COL])

    # Filter short time series
    min_required_points = MAX_ENCODER_LENGTH + MAX_PREDICTION_LENGTH
    filtered_df = df.groupby("bus_stop_id").filter(lambda x: len(x) >= min_required_points)

    if filtered_df.empty:
        raise ValueError("After filtering, no groups remain. Reduce MAX_ENCODER_LENGTH and MAX_PREDICTION_LENGTH.")

    # Train-test split
    training_cutoff = int(0.8 * len(filtered_df))
    train_df = filtered_df.iloc[:training_cutoff]
    test_df = filtered_df.iloc[training_cutoff:]

    # Feature engineering
    for split_df in [train_df, test_df]:
        split_df["hour"] = split_df[DATE_COL].dt.hour
        split_df["day_of_week"] = split_df[DATE_COL].dt.dayofweek

    # Define TimeSeriesDataSet
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="target_transformed",
        group_ids=group_ids,
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=group_ids,
        time_varying_known_reals=["hour", "day_of_week"],
        time_varying_unknown_reals=["target_transformed"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, test_df, min_prediction_idx=training_cutoff + 1)

    # Create DataLoaders
    train_dataloader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=2)
    val_dataloader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=2)

    # Define TemporalFusionTransformer model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-3,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    # Trainer setup
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, verbose=True, mode="min")
    lr_logger = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(
        max_epochs=EPOCHS,
        callbacks=[early_stop_callback, lr_logger],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else "auto",
        gradient_clip_val=0.1,
    )

    # Train the model
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Predict using a prediction DataLoader
    prediction_dataloader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=2)
    predictions = trainer.predict(tft, dataloaders=prediction_dataloader)
    predictions = torch.cat(predictions).numpy()
    predictions = np.expm1(predictions)  # Invert log1p transform

    # Extract actual values
    y_actual = test_df["target_transformed"].iloc[-len(predictions):].values
    y_actual = np.expm1(y_actual)

    # Calculate metrics
    mae = mean_absolute_error(y_actual, predictions)
    mse = mean_squared_error(y_actual, predictions)
    rmse = np.sqrt(mse)
    mask = y_actual != 0
    mape = np.mean(np.abs((y_actual[mask] - predictions[mask]) / y_actual[mask])) * 100 if mask.any() else np.nan

    print("TFT MAE:", mae)
    print("TFT RMSE:", rmse)
    if not np.isnan(mape):
        print(f"TFT MAPE: {mape:.2f}%")

    # Save model and results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('./models', exist_ok=True)
    model_filename = f'./models/tft_{timestamp}.ckpt'
    trainer.save_checkpoint(model_filename)

    results = {
        'model': 'TemporalFusionTransformer',
        'timestamp': timestamp,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape if not np.isnan(mape) else None
    }

    os.makedirs('./results', exist_ok=True)
    results_filename = f'./results/tft_results_{timestamp}.json'
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Model saved to {model_filename}")
    print(f"Results saved to {results_filename}")
