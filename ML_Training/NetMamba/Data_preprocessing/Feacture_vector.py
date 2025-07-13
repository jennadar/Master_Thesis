import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler

# Path to the dataset
dataset_path = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/CTU-IoT-Malware-Capture-8-1/netmamba_dataset/"
train_path = os.path.join(dataset_path, "train")
valid_path = os.path.join(dataset_path, "valid")
test_path = os.path.join(dataset_path, "test")

# Ensure the directories exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(valid_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Fixed sequence length for padding (define based on data analysis)
MAX_SEQ_LEN = 20  # Adjust based on dataset analysis
NUM_FEATURES = 8  # Number of features per packet

# Function to extract features from the flow
def process_flow(flow):
    flow_features = []

    for packet in flow:
        packet_features = [
            packet["packet_length"],
            packet["inter_packet_time"],
            packet["flow_duration"],
            packet["total_bytes_in_flow"],
            packet["mean_packet_size"],
            packet["mean_inter_arrival_time"],
            packet["packets_per_second"],
            packet["bytes_per_second"],
        ]
        flow_features.append(packet_features)

    # Convert to numpy array
    flow_features = np.array(flow_features, dtype=np.float32)

    # Pad or truncate to fixed length
    if len(flow_features) > MAX_SEQ_LEN:
        flow_features = flow_features[:MAX_SEQ_LEN]  # Truncate
    else:
        padding = np.zeros((MAX_SEQ_LEN - len(flow_features), NUM_FEATURES))
        flow_features = np.vstack((flow_features, padding))  # Pad

    return flow_features

# Function to convert flows into feature vectors
def convert_flows_to_feature_vectors(flow_folder):
    feature_vectors = []
    labels = []

    for attack_type in os.listdir(flow_folder):
        attack_folder = os.path.join(flow_folder, attack_type)
        if not os.path.isdir(attack_folder):
            continue

        for json_file in os.listdir(attack_folder):
            if json_file.endswith(".json"):
                json_file_path = os.path.join(attack_folder, json_file)
                with open(json_file_path, 'r') as f:
                    flow_data = json.load(f)

                # Extract features and pad sequences
                flow_features = process_flow(flow_data)
                feature_vectors.append(flow_features)
                labels.append(attack_type)  # Store attack type as label

    return np.array(feature_vectors, dtype=np.float32), np.array(labels)

# Convert datasets
train_features, train_labels = convert_flows_to_feature_vectors(train_path)
valid_features, valid_labels = convert_flows_to_feature_vectors(valid_path)
test_features, test_labels = convert_flows_to_feature_vectors(test_path)

# Standardize features across the entire dataset
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features.reshape(-1, NUM_FEATURES)).reshape(train_features.shape)
valid_features = scaler.transform(valid_features.reshape(-1, NUM_FEATURES)).reshape(valid_features.shape)
test_features = scaler.transform(test_features.reshape(-1, NUM_FEATURES)).reshape(test_features.shape)

# Save preprocessed data
np.save(os.path.join(dataset_path, "train_features.npy"), train_features)
np.save(os.path.join(dataset_path, "train_labels.npy"), train_labels)
np.save(os.path.join(dataset_path, "valid_features.npy"), valid_features)
np.save(os.path.join(dataset_path, "valid_labels.npy"), valid_labels)
np.save(os.path.join(dataset_path, "test_features.npy"), test_features)
np.save(os.path.join(dataset_path, "test_labels.npy"), test_labels)

print("Feature vectors and labels saved successfully.")
