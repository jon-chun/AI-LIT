
import os
import pickle
import pandas as pd
from typing import Dict


def read_aggregate_pkl_files(directory_name: str) -> Dict[str, pd.DataFrame]:
    """
    Reads all the .pkl files in the given directory and aggregates them into a dictionary.

    Parameters:
    directory_name (str): The directory containing .pkl files.

    Returns:
    Dict[str, pd.DataFrame]: A dictionary where keys are filenames (without .pkl extension) and values are DataFrames.
    """
    aggregated_dict = {}

    if not os.path.exists(directory_name):
        raise FileNotFoundError(f"The directory {directory_name} does not exist.")
    
    for filename in os.listdir(directory_name):
        if filename.endswith('.pkl'):
            filepath = os.path.join(directory_name, filename)
            with open(filepath, 'rb') as file:
                df = pickle.load(file)
                filename_key = filename.replace('.pkl', '')
                aggregated_dict[filename_key] = df
                print(f"Loaded {filename}")

    return aggregated_dict

def stylo_dict_to_df(stylo_aggregated_dict: Dict[str, pd.DataFrame], directory_name: str) -> None:
    """
    Converts the aggregated dictionary of DataFrames into a single DataFrame and saves it as a .csv file.
    Prints the head, tail, info, describe, and a summary of the DataFrame.

    Parameters:
    stylo_aggregated_dict (Dict[str, pd.DataFrame]): The aggregated dictionary of DataFrames.
    directory_name (str): The directory where the .csv file will be saved.
    """
    # Flatten the dictionary into a single DataFrame
    df_list = []
    for filename, df in stylo_aggregated_dict.items():
        df['filename'] = filename  # Add a column for the filename
        df_list.append(df)
    combined_df = pd.concat(df_list, ignore_index=True)

    # Save the DataFrame to a .csv file
    csv_filename = os.path.join(directory_name, 'stylo_aggregated_metrics.csv')
    combined_df.to_csv(csv_filename, index=False)
    print(f"Saved CSV file to {csv_filename}")

    # Print the DataFrame summaries
    print("DataFrame Head:")
    print(combined_df.head())

    print("\nDataFrame Tail:")
    print(combined_df.tail())

    print("\nDataFrame Info:")
    print(combined_df.info())

    print("\nDataFrame Describe:")
    print(combined_df.describe())

    print("\nDataFrame Summary:")
    print(combined_df.describe(include='all'))

# Example usage:
path_stylo_pkl = os.path.join("..", "data", "stylometrics", "textdescriptives")

try:
    stylo_aggregated_dict = read_aggregate_pkl_files(path_stylo_pkl)
    stylo_dict_to_df(stylo_aggregated_dict, path_stylo_pkl)
except FileNotFoundError as e:
    print(e)

# Example usage:
# directory_name = 'temp'  # The directory where your .pkl files are stored
# aggregated_metrics_dict = read_aggregate_pkl_files(directory_name)

path_stylo_pkl = os.path.join("..","data","stylometrics","textdescriptives")
stylo_aggregated_dict = read_aggregate_pkl_files(path_stylo_pkl)
print(stylo_aggregated_dict)


# Example usage:
# directory_name = os.path.join("..","data","stylometrics","textdescriptives")
# stylo_aggregated_dict = read_aggregate_pkl_files(directory_name)
stylo_dict_to_df(stylo_aggregated_dict, path_stylo_pkl)

