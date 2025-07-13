'''
# part 1:creating multiple csv of feature csv

import pandas as pd
import os

def split_csv_into_sections(input_csv, output_folder, num_sections=10):
    """Splits a CSV file into 'num_sections' equal parts and saves them in 'output_folder'."""
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the CSV file
    df = pd.read_csv(input_csv)
    total_rows = len(df)
    
    if total_rows == 0:
        print("❌ ERROR: Input CSV is empty!")
        return

    print(f"✅ Total rows: {total_rows}, Splitting into {num_sections} parts...")

    # Calculate chunk size
    chunk_size = total_rows // num_sections
    remainder = total_rows % num_sections  # Handle any remainder

    start_idx = 0
    for i in range(num_sections):
        # Ensure last section gets any remaining rows
        end_idx = start_idx + chunk_size + (1 if i < remainder else 0)

        # Extract subset and save
        chunk_df = df.iloc[start_idx:end_idx]
        #change the file_name here 
        output_file = os.path.join(output_folder, f"2018-05-09-192.168.100.103.horizontalPort_packet_flow_features_{i+1}.csv")
        chunk_df.to_csv(output_file, index=False)

        print(f" Saved: {output_file} ({len(chunk_df)} rows)")

        start_idx = end_idx  # Update for next section

    print("\n All sections saved successfully!")

# Example Usage
input_csv = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/CTU-IoT-Malware-Capture-1-1/2018-05-09-192.168.100.103.horizontalPort_packet_flow_features.csv"
output_folder = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/CTU-IoT-Malware-Capture-1-1/Feature_csv_list/"
split_csv_into_sections(input_csv, output_folder)
'''

'''#Part2
import pandas as pd
import os
from glob import glob

def parse_conn_log_labeled(log_file):
    """Parse Zeek conn.log.labeled file and extract relevant fields."""
    labels = []
    with open(log_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split()
            if len(fields) < 9:
                continue
            ts = round(float(fields[0]), 6)
            src_ip = fields[2]
            label = fields[-2]
            detailed_label = fields[-1]
            labels.append((ts, src_ip, label, detailed_label))

    return pd.DataFrame(labels, columns=["ts", "src_ip", "label", "detailed_label"])

def map_labels_to_features(feature_csv, log_file, output_csv):
    """Map labels from conn.log.labeled to features CSV while maintaining order."""
    
    df = pd.read_csv(feature_csv, dtype={"src_ip": "string", "dst_ip": "string", "proto": "string",
                                         "src_port": "int32", "dst_port": "int32"}, low_memory=False)
    
    df["ts"] = df["ts"].round(6)  # Ensure timestamp precision matches Zeek logs
    df["flow_duration"] = df["flow_duration"].astype("float32")  # Optimize memory usage

    # Load labeled log data
    labeled_df = parse_conn_log_labeled(log_file)

    # Merge based on (ts, src_ip)
    df = df.merge(labeled_df, on=["ts", "src_ip"], how="left", sort=False)

    # Fill missing labels
    df["label"].fillna("Unknown", inplace=True)
    df["detailed_label"].fillna("Unknown", inplace=True)

    # Handle unknown labels
    df = classify_unknown_labels(df)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f" Labeled file saved: {output_csv}")

def classify_unknown_labels(df):
    """Classifies unknown labels using flow_duration and previous packet labels."""
    
    mask = (df["label"] == "Unknown") & (df["flow_duration"] > 1)
    df.loc[mask, ["label", "detailed_label"]] = ["Benign", "-"]

    flow_keys = ["src_ip", "dst_ip", "proto", "src_port", "dst_port"]
    
    df.sort_values(["src_ip", "dst_ip", "proto", "src_port", "dst_port", "ts"], inplace=True)
    df["label"] = df.groupby(flow_keys)["label"].transform(lambda x: x.ffill().bfill())
    df["detailed_label"] = df.groupby(flow_keys)["detailed_label"].transform(lambda x: x.ffill().bfill())

    df["label"].replace("Unknown", "Benign", inplace=True)
    df["detailed_label"].replace("Unknown", "-", inplace=True)

    return df

def process_all_csvs(feature_folder, log_file, output_folder):
    """Processes all CSVs in 'feature_folder', labels them, and saves in 'output_folder'."""
    
    os.makedirs(output_folder, exist_ok=True)

    feature_csvs = glob(os.path.join(feature_folder, "*.csv"))

    if not feature_csvs:
        print(" No CSV files found in the feature folder!")
        return

    print(f"🔹 Found {len(feature_csvs)} feature files. Processing...")

    for feature_csv in feature_csvs:
        filename = os.path.basename(feature_csv)
        output_csv = os.path.join(output_folder, filename)

        map_labels_to_features(feature_csv, log_file, output_csv)

    print("\n All CSVs labeled and saved successfully!")

# Example Usage
feature_folder = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/CTU-IoT-Malware-Capture-1-1/Feature_csv_list"
log_file = "D:/jenny/Documents/FAUS_Study/Thesis/iot_23_datasets_full/IoTScenarios/CTU-IoT-Malware-Capture-1-1/bro/conn.log.labeled"
output_folder = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/CTU-IoT-Malware-Capture-1-1/Labeled_feature_csv"

process_all_csvs(feature_folder, log_file, output_folder)
'''

#'''#Part3
import pandas as pd
import os

def merge_csv_in_folder(input_folder, output_file):
    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
    
    # Initialize an empty list to store dataframes
    df_list = []
    
    # Loop through each CSV file and read it
    for csv_file in csv_files:
        file_path = os.path.join(input_folder, csv_file)
        df = pd.read_csv(file_path)
        
        # Append the dataframe to the list
        df_list.append(df)
    
    # Concatenate all dataframes into one
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Sort the dataframe by the 'ts' column (timestamp)
    combined_df.sort_values(by="ts", inplace=True)
    
    # Save the combined dataframe to a new CSV file
    combined_df.to_csv(output_file, index=False)
    
    print(f"All CSVs merged and sorted by 'ts'. The output is saved to: {output_file}")

# Example usage
input_folder = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/CTU-IoT-Malware-Capture-1-1/Labeled_feature_csv"
output_file = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/CTU-IoT-Malware-Capture-1-1/2018-05-09-192.168.100.103.horizontalPort_packet_flow_features_merged_labeled.csv"

merge_csv_in_folder(input_folder, output_file)
#'''