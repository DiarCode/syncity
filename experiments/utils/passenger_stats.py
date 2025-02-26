import numpy as np
import pandas as pd


def analyze_passenger_data(csv_filepath):
    """
    Analyzes passenger data from a CSV file and calculates descriptive statistics.

    Args:
        csv_filepath: Path to the CSV file.

    Returns:
        A dictionary containing descriptive statistics for key variables,
        or an error message string if an error occurs.
    """
    try:
        df = pd.read_csv(csv_filepath)

        # Convert to Datetime (handling errors)
        df['bus_board_computer_sent_time'] = pd.to_datetime(
            df['bus_board_computer_sent_time'], errors='coerce')
        df['created_time'] = pd.to_datetime(
            df['created_time'], errors='coerce')

        # Drop rows with NaT (invalid datetime)
        df = df.dropna(subset=['bus_board_computer_sent_time', 'created_time'])

        # Feature Engineering (net_passenger_change)
        df['net_passenger_change'] = df['enter_sum'] - df['exit_sum']

        # Key variables for analysis
        key_variables = ['enter_sum', 'exit_sum',
                         'tickets_count', 'net_passenger_change']

        # Calculate descriptive statistics
        descriptive_stats = df[key_variables].describe()

        # Calculate median separately (as .describe() doesn't include it by default for numeric columns)
        median_values = df[key_variables].median()
        # Add median to the stats table
        descriptive_stats.loc['median'] = median_values

        return {"descriptive_stats": descriptive_stats}

    except FileNotFoundError:
        return f"Error: File not found: {csv_filepath}"
    except pd.errors.ParserError:
        return f"Error: Could not parse CSV file: {csv_filepath}. Check the format."
    except KeyError as e:
        return f"Error: Column '{e}' not found in CSV."
    except Exception as e:
        return f"An unexpected error occurred: {e}"


# Example usage (in a separate file, e.g., analyze_script.py):
filepath = './data/passflow.csv'  # Or the correct path to your CSV
results = analyze_passenger_data(filepath)

if isinstance(results, dict):
    print("Descriptive Statistics:\n", results['descriptive_stats'])

elif isinstance(results, str):
    print(results)  # print error message    print(results) #print error message
