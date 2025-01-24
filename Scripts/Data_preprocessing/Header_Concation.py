import pandas as pd
import os

# Define input and output directories
input_dir = r"D:/jenny/Documents/FAUS_Study/Thesis/My_dataset/Raw_data/"
output_dir = r"D:/jenny/Documents/FAUS_Study/Thesis/My_dataset/Raw_data/"
features_file = r"D:/jenny/Documents/FAUS_Study/Thesis/UNSW-NB15/CSV Files/NUSW-NB15_features.csv"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the features file
features = pd.read_csv(features_file, encoding="latin1")

# Extract the feature names
column_names = features["Name"].tolist()

# Specify the files to process
files_to_process = ["UNSW-NB15_1.csv", "UNSW-NB15_2.csv", "UNSW-NB15_3.csv", "UNSW-NB15_4.csv"]

# Process each specified file
for file_name in files_to_process:
    input_file = os.path.join(input_dir, file_name)
    output_file = os.path.join(output_dir, file_name)
    
    # Check if the input file exists
    if os.path.exists(input_file):
        # Load the data file
        data = pd.read_csv(input_file, header=None)
        
        # Add headers
        data.columns = column_names

        # Save the updated file
        data.to_csv(output_file, index=False)
        print(f"Processed and saved: {output_file}")
    else:
        print(f"File not found: {input_file}")

print("Processing complete for specified files.")
