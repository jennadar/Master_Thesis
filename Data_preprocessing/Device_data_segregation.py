# Data segration depending on Srcip for creating local devices.
import pandas as pd
import os

# Input file containing normal data
input_file = "D:/jenny/Documents/FAUS_Study/Thesis/My_dataset/Test_data/Nor80_back20/test_data_n80_backdoor20.csv"  # Replace with the path to your normal data file
output_dir = "D:/jenny/Documents/FAUS_Study/Thesis/My_dataset/Test_data/Nor80_back20/Device_data/"  # Directory to store the segregated files

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
data = pd.read_csv(input_file)

# Ensure the 'srcip' column exists
if "srcip" not in data.columns:
    raise ValueError("The dataset does not contain the 'srcip' column.")

# Get the unique source IPs
unique_srcips = data["srcip"].unique()

# Loop through each unique source IP, filter data, and save to a CSV
for idx, srcip in enumerate(unique_srcips, start=1):
    # Filter data for the current source IP
    device_data = data[data["srcip"] == srcip]
    
    # Define the output file name
    output_file = os.path.join(output_dir, f"device_{idx}_data.csv")
    
    # Save the data to the CSV
    device_data.to_csv(output_file, index=False)
    print(f"Data for srcip '{srcip}' saved to {output_file}")

print(f"Data segregation completed. Files are saved in the '{output_dir}' directory.")
