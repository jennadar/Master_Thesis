'''
******************************************************************************************************

Distinguish between normal and attacked data

******************************************************************************************************
'''

import pandas as pd
import os

# Define input directory (where modified files are stored) and output file paths
input_dir = r"D:/jenny/Documents/FAUS_Study/Thesis/My_dataset/Raw_data/"
output_normal_file = r"D:/jenny/Documents/FAUS_Study/Thesis/My_dataset/Normal_data/UNSW-NB15_normal.csv"
output_attack_file = r"D:/jenny/Documents/FAUS_Study/Thesis/My_dataset/Attacked_data/UNSW-NB15_attack.csv"

# Initialize empty dataframes for normal and attack datasets
normal_df = pd.DataFrame()
attack_df = pd.DataFrame()

# List all CSV files in the input directory
data_files = [file for file in os.listdir(input_dir) if file.endswith(".csv")]

# Process each file
for file_name in data_files:
    file_path = os.path.join(input_dir, file_name)
    print(f"Processing file: {file_path}")
    
    # Load the file
    data = pd.read_csv(file_path)
    
    # Check if the required 'label' column exists
    if "Label" in data.columns:
        # Split into normal and attack data
        normal_data = data[data["Label"] == 0]
        attack_data = data[data["Label"] == 1]
        
        # Append to the combined normal and attack datasets
        normal_df = pd.concat([normal_df, normal_data], ignore_index=True)
        attack_df = pd.concat([attack_df, attack_data], ignore_index=True)
    else:
        print(f"Warning: 'label' column not found in file {file_name}. Skipping.")

# Save the final datasets
normal_df.to_csv(output_normal_file, index=False)
attack_df.to_csv(output_attack_file, index=False)

print(f"Normal data saved to: {output_normal_file}")
print(f"Attack data saved to: {output_attack_file}")
