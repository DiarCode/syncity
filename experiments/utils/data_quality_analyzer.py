import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def detect_outliers_bus_data(df, passenger_col, window=15, std_threshold=3):
    """Detects outliers in bus passenger data using STL and rolling statistics."""
    try:
        df = df.set_index('created_time')
    except KeyError:
        return "Error: 'created_time' column not found."
    except TypeError:
        return "Error: 'created_time' column is not appropriate for index setting"

    try:
        stl = STL(df[passenger_col], period=24)  # Assuming daily seasonality
        stl_result = stl.fit()
        residuals = stl_result.resid

        rolling_mean = residuals.rolling(window=window, center=True).mean()
        rolling_std = residuals.rolling(window=window, center=True).std()

        outliers = np.abs((residuals - rolling_mean) /
                          rolling_std) > std_threshold
        df['is_outlier'] = outliers

        return df
    except Exception as e:
        return f"An error occurred during outlier detection: {e}"


def check_data_consistency(df, enter_col='enter_sum', exit_col='exit_sum', tickets_col='tickets_count'):
    """Checks for data consistency issues like negative counts or inconsistencies between columns."""

    inconsistencies = []

    # Check for negative counts
    negative_enter = df[df[enter_col] < 0]
    if not negative_enter.empty:
        inconsistencies.append(
            f"Found {len(negative_enter)} instances of negative {enter_col}.")

    negative_exit = df[df[exit_col] < 0]
    if not negative_exit.empty:
        inconsistencies.append(
            f"Found {len(negative_exit)} instances of negative {exit_col}.")

    negative_tickets = df[df[tickets_col] < 0]
    if not negative_tickets.empty:
        inconsistencies.append(
            f"Found {len(negative_tickets)} instances of negative {tickets_col}.")

    # Check for inconsistencies between enter, exit, and tickets (example logic)
    # More sophisticated logic might be needed based on the specific meaning of your data
    inconsistent_rows = df[df[tickets_col] > (df[enter_col] + df[exit_col])]
    if not inconsistent_rows.empty:
        inconsistencies.append(f"Found {len(inconsistent_rows)} rows where {
                               tickets_col} is greater than the sum of {enter_col} and {exit_col}.")

    return inconsistencies


def analyze_csv_with_outliers(csv_filepath, passenger_column='enter_sum', enter_col='enter_sum', exit_col='exit_sum', tickets_col='tickets_count'):
    """Imports CSV, detects outliers, checks for inconsistencies, and returns results."""
    try:
        df = pd.read_csv(csv_filepath, parse_dates=['created_time'])
    except FileNotFoundError:
        return "Error: File not found."
    except pd.errors.ParserError:
        return "Error: Could not parse CSV file. Check the format."
    except KeyError as e:
        return f"Error: Column '{e}' not found in CSV."
    except Exception as e:
        return f"An unexpected error occurred during CSV import: {e}"

    consistency_errors = check_data_consistency(
        df, enter_col, exit_col, tickets_col)

    df_with_outliers = detect_outliers_bus_data(df, passenger_column)

    if isinstance(df_with_outliers, str):
        return df_with_outliers

    outliers_df = df_with_outliers[df_with_outliers['is_outlier']]
    outliers_count = len(outliers_df)
    passenger_stats = df[passenger_column].describe()

    return {
        "df_with_outliers": df_with_outliers,
        "outliers_df": outliers_df,
        "outliers_count": outliers_count,
        "passenger_stats": passenger_stats,
        "consistency_errors": consistency_errors
    }


filepath = './data/passflow.csv'
results = analyze_csv_with_outliers(filepath)

if isinstance(results, dict):
    print("DataFrame with Outliers (first 5 rows):\n",
          results['df_with_outliers'].head())
    print("\nOutliers Only:\n", results['outliers_df'])
    print(f"\nTotal Outliers Count: {results['outliers_count']}")
    print("\nPassenger Statistics:\n", results['passenger_stats'])
    if results.get("consistency_errors"):
        print("Data Consistency Issues:")
        for error in results["consistency_errors"]:
            print(error)
elif isinstance(results, str):
    print(results)
