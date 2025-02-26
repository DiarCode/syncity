import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
data = pd.read_csv('./data/passflow.csv')

# Ensure datetime conversion
data['bus_board_computer_sent_time'] = pd.to_datetime(
    data['bus_board_computer_sent_time'], errors='coerce')

# Extract the hour from the timestamp
data['hour'] = data['bus_board_computer_sent_time'].dt.hour

# Group by hour and calculate total enter_sum and exit_sum
hourly_flow = data.groupby(
    'hour')[['enter_sum', 'exit_sum']].sum().reset_index()

# Plotting peak hour identification
plt.figure(figsize=(12, 6))
plt.plot(hourly_flow['hour'], hourly_flow['enter_sum'],
         label='Enter Sum', marker='o')
plt.plot(hourly_flow['hour'], hourly_flow['exit_sum'],
         label='Exit Sum', marker='o')

# Add titles and labels
plt.title('Peak Hour Identification - Passenger Flow', fontsize=14)
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Total Passenger Flow', fontsize=12)
plt.xticks(range(0, 24))  # Ensure all hours are displayed on the x-axis
plt.legend(fontsize=10)
plt.grid(alpha=0.3)

# Show the plot
plt.tight_layout()
plt.show()
