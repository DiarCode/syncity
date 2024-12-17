import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Step 1: Load the Data
# If you are sure it's tab-separated, keep sep='\t'. Otherwise, try sep=',' if your CSV is comma-separated.
df = pd.read_csv('./data/passflow.csv')

# Step 2: Convert to Datetime
df['bus_board_computer_sent_time'] = pd.to_datetime(
    df['bus_board_computer_sent_time'], errors='coerce')
df['created_time'] = pd.to_datetime(df['created_time'], errors='coerce')

# Drop rows with NaT (invalid datetime) to ensure all datetimes are valid
df = df.dropna(subset=['bus_board_computer_sent_time', 'created_time'])

# Verify that datetime conversion worked
if not np.issubdtype(df['bus_board_computer_sent_time'].dtype, np.datetime64):
    raise ValueError(
        "bus_board_computer_sent_time was not converted to datetime. Check data formatting.")

if not np.issubdtype(df['created_time'].dtype, np.datetime64):
    raise ValueError(
        "created_time was not converted to datetime. Check data formatting.")

# Step 3: Basic EDA and Data Checks
print("Data Head:")
print(df.head())
print("\nInfo:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

print("\nDescriptive Statistics:")
print(df.describe(include='all'))

print("\nUnique route numbers:", df['route_number'].unique())
print("Unique bus_stop_ids:", df['bus_stop_id'].nunique())
print("Unique bus_ids:", df['bus_id'].nunique())

# Distributions
plt.figure(figsize=(8, 5))
sns.histplot(df['enter_sum'], kde=True)
plt.title("Distribution of Enter Sum")
plt.xlabel("enter_sum")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df['exit_sum'], kde=True)
plt.title("Distribution of Exit Sum")
plt.xlabel("exit_sum")
plt.show()

# Relationship between tickets_count and enter_sum
plt.figure(figsize=(8, 5))
sns.scatterplot(x='tickets_count', y='enter_sum', data=df)
plt.title("Tickets Count vs Enter Sum")
plt.show()

# Step 4: Handle Missing Values
# tickets_count has some missing values - only a few missing out of total rows.
# We'll fill them with the median for simplicity.
df['tickets_count'] = df['tickets_count'].fillna(df['tickets_count'].median())

# Step 5: Feature Engineering
# Extract time-based features from bus_board_computer_sent_time
df['hour'] = df['bus_board_computer_sent_time'].dt.hour
df['day_of_week'] = df['bus_board_computer_sent_time'].dt.dayofweek
df['date'] = df['bus_board_computer_sent_time'].dt.date

# Create net_passenger_change
df['net_passenger_change'] = df['enter_sum'] - df['exit_sum']

# Sort data for lag features
df = df.sort_values(by=['route_number', 'bus_stop_id',
                    'bus_board_computer_sent_time'])

# Create lag features
group_cols = ['route_number', 'bus_stop_id']
df['enter_sum_lag1'] = df.groupby(group_cols)['enter_sum'].shift(1)
df['exit_sum_lag1'] = df.groupby(group_cols)['exit_sum'].shift(1)
df['tickets_lag1'] = df.groupby(group_cols)['tickets_count'].shift(1)

# Rolling mean (transform ensures alignment of indices)
df['enter_rolling_mean_3'] = df.groupby(group_cols)['enter_sum'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean())
df['exit_rolling_mean_3'] = df.groupby(group_cols)['exit_sum'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean())
df['tickets_rolling_mean_3'] = df.groupby(group_cols)['tickets_count'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean())

# Drop rows with NaN introduced by lagging (first records in each group)
df = df.dropna(subset=['enter_sum_lag1', 'exit_sum_lag1', 'tickets_lag1'])

# Step 6: Correlation Analysis
cor_features = ['enter_sum', 'enter_sum_lag1', 'exit_sum_lag1', 'tickets_lag1',
                'enter_rolling_mean_3', 'exit_rolling_mean_3', 'tickets_rolling_mean_3',
                'hour', 'day_of_week', 'net_passenger_change']
corr = df[cor_features].corr()
print("\nCorrelation with enter_sum:")
print(corr['enter_sum'].sort_values(ascending=False))

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# After EDA & FE, we have a more refined dataset ready for modeling.
df.to_csv('./data/passflow_prepared.csv', index=False)
