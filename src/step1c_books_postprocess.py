# SETUP AND CONFIGURATION
import os
import re
import shutil
import string
import codecs
import logging


# Define diretory by DEBUG Mode
DEBUG_FLAG = False

if DEBUG_FLAG:
    # Debug Mode
    SUBDIR_SEGMENT = os.path.join("data", "step1_segments")
else:
    # Normal Execution Mode
    SUBDIR_SEGMENT = os.path.join("..", "data", "step1_segments")

# Rule to filter out lines
MIN_LENGTH = 3


# COMMON FUNCTIONS

def get_txt_file_paths(base_dir):
    txt_file_paths = []
    
    # Traverse the directory tree starting from the base directory
    for root, dirs, files in os.walk(base_dir):
        # Iterate over each file in the current directory
        for file in files:
            # Check if the file has a .txt extension
            if file.endswith(".txt"):
                # Construct the full relative path to the .txt file
                file_path = os.path.join(root, file)
                # Append the file path to the list
                txt_file_paths.append(file_path)
    
    return txt_file_paths


def process_text_file(file_path, min_length=3):
    # Configure logging
    logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Extract the directory and filename from the file path
    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    
    # Create the new file name by removing the "_raw.txt" suffix
    cleaned_filename = filename.replace("_raw.txt", ".txt")
    cleaned_file_path = os.path.join(directory, cleaned_filename)

    # Open the file for reading
    with codecs.open(file_path, "r", encoding="utf-8", errors="replace") as file:
        # Read the entire file contents
        text = file.read()

    # Replace illegal/non-printable characters with a space
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)

    # Replace multiple consecutive newline characters with two newline characters
    text = re.sub(r"\n{2,}", "\n\n", text)

    # Replace multiple consecutive spaces with a single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()
        
    # Iterate over each line in the input file
    paragraphs_list = re.split(r"\n{2,}", text)

    # First Join all lines into one text string
    for i_index, paragraph_now in enumerate(paragraphs_list):
        paragraph_clean = (" ").join(paragraph_now.split())
        paragraphs_list[i_index] = paragraph_clean


    # Split it into a list of paragraphs (boundry 2+ all whitespace lines)

    # (" ").join(each paragraph) each paragrah individually

    # ("\n\n").join(all paragraphs)


        print(f"type(file_in): {type(file_in)}")
        for line_num, line in enumerate(file_in, start=1):
            try:
                # Strip leading/trailing whitespace from the line
                line = line.strip()
                
                # Check if the line meets the filtering criteria
                if (
                    len(line) >= min_length and  # Line length is greater than or equal to MIN_LENGTH
                    not line.isspace() and  # Line is not only whitespace
                    not all(char in string.punctuation for char in line)  # Line is not only punctuation
                ):
                    # Write the filtered line to the output file
                    file_out.write(line + "\n")
            except UnicodeDecodeError as e:
                logging.error(f"UnicodeDecodeError in file: {file_path}, line: {line_num}, error: {e}")
                print(f"Skipping line {line_num} in file: {file_path} due to UnicodeDecodeError: {e}")
            except Exception as e:
                logging.error(f"Unexpected error in file: {file_path}, line: {line_num}, error: {e}")
                print(f"Skipping line {line_num} in file: {file_path} due to unexpected error: {e}")
    
    print(f"Processed file: {file_path}")
    print(f"Cleaned file created: {cleaned_file_path}")



# MAIN LOOP

# Check DEBUG_FLAG for correct directory paths
subdir_book_segments = []

fullpath_segment_files_list = get_txt_file_paths(SUBDIR_SEGMENT)

print(fullpath_segment_files_list)

for i_index, fullpath_segment_file in enumerate(fullpath_segment_files_list):
    print(f"PROCESSING segment file #{i_index}: {fullpath_segment_file}")
    process_text_file(fullpath_segment_file)
