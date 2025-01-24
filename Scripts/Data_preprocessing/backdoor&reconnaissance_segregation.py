import pandas as pd
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
backdoor_file = r"D:/jenny/Documents/FAUS_Study/Thesis/My_dataset/Attacked_data/UNSW-NB15_backdoor_only.csv"
rec_file = r"D:/jenny/Documents/FAUS_Study/Thesis/My_dataset/Attacked_data/UNSW-NB15_reconnaissance_only.csv"

# Load the data
data = pd.read_csv(output_attack_file)

# Debugging: Display column names and unique values in attack_cat
print(f"Columns in data: {data.columns}")
if "attack_cat" in data.columns:
    print(f"Unique values in 'attack_cat': {data['attack_cat'].unique()}")
else:
    print("Column 'attack_cat' not found!")

# Clean column names and values if necessary
data.columns = data.columns.str.strip()  # Remove spaces from column names
data["attack_cat"] = data["attack_cat"].astype(str).str.strip()  # Clean the 'attack_cat' values

# Filter data
backdoor_attack_data = data[data["attack_cat"] == "Backdoors"]
print(f"Backdoor data count: {len(backdoor_attack_data)}")
backdoor_attack_data.to_csv(backdoor_file, index=False)
print(backdoor_attack_data)  # Display filtered rows

reconnaissance_attack_data = data[data["attack_cat"] == "Reconnaissance"]
print(f"Reconnaissance data count: {len(reconnaissance_attack_data)}")
reconnaissance_attack_data.to_csv(rec_file, index=False)
print(reconnaissance_attack_data)