#3step:
import os
import json
import pandas as pd
import random
from collections import defaultdict

# Load extracted packet flow features
csv_file = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/CTU-IoT-Malware-Capture-20-1/2018-10-02-13-12-30-192.168.100.103_packet_flow_features_labeled.csv"  # Update with the actual CSV path
df = pd.read_csv(csv_file)

# Define the dataset output structure
dataset_path = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/CTU-IoT-Malware-Capture-20-1/netmamba_dataset/"  # Update path if needed
train_path = os.path.join(dataset_path, "train")
valid_path = os.path.join(dataset_path, "valid")
test_path = os.path.join(dataset_path, "test")

# Ensure directories exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(valid_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Initialize flow storage
flow_dict = defaultdict(list)

# Define attack type mapping (can be updated with more attacks)
attack_mapping = {
    "C&C-Torii": "C&C-Torii",
    "Benign": "Benign",
    # Add more attack types if needed
}

# Iterate over each packet row in the CSV
for _, row in df.iterrows():
    flow_key = (row["src_ip"], row["dst_ip"], row["proto"], row["src_port"], row["dst_port"])
    
    # Extract attack type from the `detailed-label` column
    detailed_label = str(row["detailed_label"]).lower()
    
    # Assign attack type based on known mapping
    if detailed_label.strip() == "-":
        attack_type = "Benign"
    else:
        attack_type = "Unknown"  # Default label if no match
        for attack_name in attack_mapping:
            if attack_name.lower() in detailed_label:
                attack_type = attack_name
                break
    
    # Append packet details to its corresponding flow
    flow_dict[flow_key].append({
        "timestamp": row["ts"],
        "packet_length": row["packet_length"],
        "inter_packet_time": row["inter_packet_time"],
        "flow_duration": row["flow_duration"],
        "total_packets_in_flow": row["total_packets_in_flow"],
        "total_bytes_in_flow": row["total_bytes_in_flow"],
        "mean_packet_size": row["mean_packet_size"],
        "mean_inter_arrival_time": row["mean_inter_arrival_time"],
        "packets_per_second": row["packets_per_second"],
        "bytes_per_second": row["bytes_per_second"],
        "attack_type": attack_type,  # Store the attack type in flow metadata
    })

# Function to split data into train, valid, and test sets
def split_data(flow_dict, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    # Make sure the sum of ratios is 1
    assert train_ratio + valid_ratio + test_ratio == 1, "The sum of ratios must equal 1"
    
    # Create a list of all flows and shuffle them
    all_flows = list(flow_dict.items())
    random.shuffle(all_flows)
    
    # Split the data
    total_flows = len(all_flows)
    train_end = int(train_ratio * total_flows)
    valid_end = int((train_ratio + valid_ratio) * total_flows)
    
    train_flows = all_flows[:train_end]
    valid_flows = all_flows[train_end:valid_end]
    test_flows = all_flows[valid_end:]
    
    return train_flows, valid_flows, test_flows

# Split the data into train, valid, and test
train_flows, valid_flows, test_flows = split_data(flow_dict)

# Function to save flows to corresponding folders
def save_flows(flows, folder_path):
    for i, (flow_key, packets) in enumerate(flows):
        attack_type = packets[0]["attack_type"]  # Use 'detailed_label' to get the attack type
        
        # Define the save path based on the attack type
        flow_folder = os.path.join(folder_path, attack_type)
        os.makedirs(flow_folder, exist_ok=True)
        
        # Save each flow as a JSON file
        json_filename = os.path.join(flow_folder, f"flow_{i}.json")
        with open(json_filename, "w") as f:
            json.dump(packets, f, indent=4)

        print(f"Saved: {json_filename}")

# Save the flows into respective folders
save_flows(train_flows, train_path)
save_flows(valid_flows, valid_path)
save_flows(test_flows, test_path)
