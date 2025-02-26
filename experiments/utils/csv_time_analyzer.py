import pandas as pd


def analyze_and_sort_csv(csv_filepath):
    """Analyzes a CSV file, sorts by created_time, calculates time resolution, etc."""
    try:
        df = pd.read_csv(csv_filepath)

        # Convert datetime columns, handling various formats
        df['created_time'] = pd.to_datetime(
            df['created_time'], errors='coerce', infer_datetime_format=True)
        df['bus_board_computer_sent_time'] = pd.to_datetime(
            df['bus_board_computer_sent_time'], errors='coerce', infer_datetime_format=True)

        # Drop rows where datetime conversion failed (important!)
        df.dropna(subset=['created_time',
                  'bus_board_computer_sent_time'], inplace=True)

        # Sort by 'created_time' descending
        df_sorted = df.sort_values(by='created_time', ascending=False)

        # Calculate time differences in seconds
        df['time_diff'] = (
            df['created_time'] - df['bus_board_computer_sent_time']).dt.total_seconds()

        # Determine time resolution
        time_diffs = df['time_diff'].unique()
        if all(diff.is_integer() for diff in time_diffs):  # check if all differences are integers
            time_resolution_seconds = int(min(time_diffs))
        else:
            time_resolution_seconds = min(time_diffs)

        if time_resolution_seconds >= 60:
            time_resolution = f"{time_resolution_seconds // 60} minutes"
        elif time_resolution_seconds == 1:
            time_resolution = "seconds"
        elif time_resolution_seconds < 1 and time_resolution_seconds > 0:
            time_resolution = "milliseconds"
        elif time_resolution_seconds == 0:
            time_resolution = "less than a second"  # or instant
        else:
            time_resolution = f"{time_resolution_seconds} seconds"

        # Calculate Averages
        avg_enter_sum = df['enter_sum'].mean()
        avg_exit_sum = df['exit_sum'].mean()
        avg_tickets_count = df['tickets_count'].mean()

        return {
            "sorted_df": df_sorted,
            "time_resolution": time_resolution,
            "avg_enter_sum": avg_enter_sum,
            "avg_exit_sum": avg_exit_sum,
            "avg_tickets_count": avg_tickets_count
        }

    except FileNotFoundError:
        return "Error: File not found."
    except pd.errors.ParserError:
        return "Error: Could not parse CSV file. Check the format."
    except KeyError as e:
        return f"Error: Column '{e}' not found in CSV."
    except Exception as e:
        return f"An unexpected error occurred: {e}"


# Example usage (importing the CSV):
filepath = './data/passflow.csv'  # Replace 'your_data.csv' with the actual path
results = analyze_and_sort_csv(filepath)

if isinstance(results, dict):  # check if result is dictionary or string(error message)
    print("Sorted DataFrame (first 5 rows):\n", results['sorted_df'].head())
    print("\nTime Resolution:", results['time_resolution'])
    print("\nAverage enter_sum:", results['avg_enter_sum'])
    print("Average exit_sum:", results['avg_exit_sum'])
    print("Average tickets_count:", results['avg_tickets_count'])

elif isinstance(results, str):
    print(results)  # print error message    print(results) #print error message
