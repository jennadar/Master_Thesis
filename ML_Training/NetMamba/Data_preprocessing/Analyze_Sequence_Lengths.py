import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Path to the dataset (change if needed)
dataset_path = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/CTU-IoT-Malware-Capture-8-1/netmamba_dataset/"
train_path = os.path.join(dataset_path, "train")
valid_path = os.path.join(dataset_path, "valid")
test_path = os.path.join(dataset_path, "test")

# Function to get flow lengths from a directory
def get_flow_lengths(folder_path):
    flow_lengths = []
    
    for attack_type in os.listdir(folder_path):
        attack_folder = os.path.join(folder_path, attack_type)

        if not os.path.isdir(attack_folder):
            continue  # Skip non-folder items

        for json_file in os.listdir(attack_folder):
            if json_file.endswith(".json"):
                json_file_path = os.path.join(attack_folder, json_file)

                with open(json_file_path, 'r') as f:
                    flow_data = json.load(f)

                # Flow length = number of packets in the flow
                flow_length = len(flow_data)
                flow_lengths.append(flow_length)

    return flow_lengths

# Get sequence lengths from train, valid, and test sets
train_lengths = get_flow_lengths(train_path)
valid_lengths = get_flow_lengths(valid_path)
test_lengths = get_flow_lengths(test_path)

# Combine all lengths
all_lengths = train_lengths + valid_lengths + test_lengths

# Compute statistics
min_len = np.min(all_lengths)
max_len = np.max(all_lengths)
mean_len = np.mean(all_lengths)
median_len = np.median(all_lengths)

print(f"Flow Length Stats:")
print(f"- Min: {min_len}")
print(f"- Max: {max_len}")
print(f"- Mean: {mean_len:.2f}")
print(f"- Median: {median_len}")

# Plot histogram
plt.figure(figsize=(12, 6))
plt.hist(all_lengths, bins=50, edgecolor='black', log=True)  # Log scale to handle large variation
plt.xlabel("Number of Packets in Flow")
plt.ylabel("Log Frequency")
plt.title("Log-Scaled Distribution of Flow Lengths")
plt.grid(True)
plt.show()
