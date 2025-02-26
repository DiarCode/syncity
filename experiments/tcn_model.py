import matplotlib.pyplot as plt
import numpy as np

# Model names and their MAPE values
models = [
    "ARIMA", "LSTM", "XGBRegressor",
    "RandomForestRegressor", "LightGBMRegressor",
    "LinearRegression", "GradientBoostingRegressor"
]
mape_values = [72, 111, 37.2, 35, 34.8, 26.59, 2.21]

# Calculate accuracy as (100 - MAPE)
accuracy = [100 - mape for mape in mape_values]

# Sort models by accuracy
sorted_indices = np.argsort(accuracy)
models_sorted = [models[i] for i in sorted_indices]
accuracy_sorted = [accuracy[i] for i in sorted_indices]

# Plot the bar chart
plt.figure(figsize=(12, 8))  # Increased figure size for better fit
bars = plt.barh(models_sorted, accuracy_sorted,
                color='skyblue', edgecolor='black', alpha=0.8)

# Add accuracy labels to each bar
for bar, value in zip(bars, accuracy_sorted):
    plt.text(value + 1, bar.get_y() + bar.get_height()/2, f"{value:.2f}%",
             va='center', fontsize=10, color='black')

# Configure the plot
plt.xlabel("Accuracy (%)", fontsize=14)
plt.ylabel("Models", fontsize=14)
plt.title("Model Accuracy Comparison (Sorted by Accuracy)", fontsize=16)
plt.xlim(0, 105)  # Adjusted x-axis limits for better text placement
plt.grid(axis="x", linestyle="--", alpha=0.7)

# Adjust layout for better text fit
plt.tight_layout()
plt.show()
