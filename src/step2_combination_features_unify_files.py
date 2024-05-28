import os
import re
import shutil
import logging
import sys
import time
from tqdm import tqdm
import pandas as pd

# Number of model run per book
MODEL_COUNT=4

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_filename_root(filename):
    """
    Extract the filename root based on the first 6 parts split by '_'.
    """
    return "_".join(filename.split("_")[:6])

def combine_files_into_dataframe(file_list, directory_path):
    """
    Combine the content of files into a Pandas DataFrame.
    """
    data_dict = {}
    for file_index, file in enumerate(file_list):
        print(f"    COMBINING #{file_index} file: {file}")
        filepath = os.path.join(directory_path, file)
        if os.path.isfile(filepath):
            column_name = file.split("_")[-1].split(".")[0]  # Use the part before the file extension as the column name
            print(f"      column_name = {column_name}")
            try:
                with open(filepath, 'r', encoding='utf-8') as file_obj:
                    print(f"      READ file: {file}")
                    lines_list = file_obj.readlines()
                    data_dict[column_name] = [float(line.strip()) for line in lines_list]  # Convert values to floats
                    print(f"        Length: {len(data_dict[column_name])}")
            except IOError as e:
                print(f"      FAIL to read file: {file}")
                logging.error(f"Error reading file {filepath}: {str(e)}")
    
    df = pd.DataFrame.from_dict(data_dict)
    print(f"      RETURNING combine_files_into_dataframe() with columns: {df.columns.values.tolist()}")
    print(f"        df.head(): {df.head()}")
    return df

def process_files(directory_path, output_directory):
    """
    Process files in the specified directory.
    """
    if not os.path.exists(directory_path):
        logging.error(f"Directory {directory_path} does not exist.")
        return

    # Step 1: Read all files and sort them
    all_files = sorted([f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))])

    # Group files by their root name
    file_groups = {}
    for file in all_files:
        filename_root = get_filename_root(file)
        if filename_root not in file_groups:
            file_groups[filename_root] = []
        file_groups[filename_root].append(file)

    # Step 2: Combine every 3 files into a DataFrame
    for filename_root, files in file_groups.items():
        print(f"  PROCESSING ROOT: {filename_root}")
        logging.info(f"Processing files with root: {filename_root}")
        for i in range(0, len(files), MODEL_COUNT):
            subset_files = files[i:i+MODEL_COUNT]
            print(f"  COMBINING {len(subset_files)} files: {subset_files}")
            df = combine_files_into_dataframe(subset_files, directory_path)
            print(f"    BACK in process_files() with df: {df.head()}")
            # Step 3: Create output filename
            output_filename = filename_root + "_sentiment_combined.csv"
            output_filepath = os.path.join(output_directory, output_filename)

            # Step 4: Write DataFrame to output file
            try:
                os.makedirs(output_directory, exist_ok=True)  # Create the output directory if it doesn't exist
                df.to_csv(output_filepath, index=False)
                logging.info(f"Combined file written to {output_filepath}")
                # return True
            
            except IOError as e:
                logging.error(f"Error writing file {output_filepath}: {str(e)}")
                # return False
            

if __name__ == "__main__":
    # Specify the input and output directory paths
    PATH_FILES_INPUT = os.path.join('..', 'data', 'step2_combination_features')
    PATH_FILES_OUTPUT = os.path.join('..', 'data', 'step3_analysis')

    # Process the files
    if process_files(PATH_FILES_INPUT, PATH_FILES_OUTPUT):
        print(f"SUCCESS: process_files() succeeded")
    else:
        print(f"FAILURE: process_files() failed")
