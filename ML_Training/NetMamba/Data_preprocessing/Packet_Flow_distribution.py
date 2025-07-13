import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Define paths
dataset_path = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/CTU-IoT-Malware-Capture-20-1/netmamba_dataset/"
train_path = os.path.join(dataset_path, "train")
valid_path = os.path.join(dataset_path, "valid")
test_path = os.path.join(dataset_path, "test")

# Step 1: Load Flow Packets
def load_flow_packets(flow_file):
    """Load packets from a flow JSON file."""
    with open(flow_file, 'r') as f:
        flow_data = json.load(f)
    
    packets = []
    for packet in flow_data:
        raw_bytes = bytes.fromhex(packet["raw_payload"])  # Convert HEX string to bytes
        packets.append(raw_bytes)
    
    return packets

# Step 2: Analyze Packet Lengths and Flow Lengths
def analyze_packet_and_flow_lengths(flow_folder):
    """Analyze packet lengths and flow lengths to determine max values."""
    packet_lengths = []
    flow_lengths = []

    for attack_type in os.listdir(flow_folder):
        attack_folder = os.path.join(flow_folder, attack_type)
        if not os.path.isdir(attack_folder):
            continue

        for json_file in os.listdir(attack_folder):
            if json_file.endswith(".json"):
                json_file_path = os.path.join(attack_folder, json_file)
                
                # Step 1: Load flow packets
                packets = load_flow_packets(json_file_path)
                
                # Analyze packet lengths
                for packet in packets:
                    packet_lengths.append(len(packet))
                
                # Analyze flow length (sum of packet lengths)
                flow_lengths.append(sum(len(packet) for packet in packets))
    
    return packet_lengths, flow_lengths

# Step 3: Determine Max Packet Length, Max Flow Length, and Stride Size
def determine_parameters():
    """Determine the max packet length, max flow length, and stride size."""
    # Analyze dataset
    packet_lengths, flow_lengths = analyze_packet_and_flow_lengths(train_path)

    # Determine MAX_PACKET_LEN (95th percentile of packet lengths)
    max_packet_len = np.percentile(packet_lengths, 95)
    print(f"MAX_PACKET_LEN (95th percentile): {max_packet_len}")

    # Determine MAX_FLOW_LEN (95th percentile of flow lengths)
    max_flow_len = np.percentile(flow_lengths, 95)
    print(f"MAX_FLOW_LEN (95th percentile): {max_flow_len}")

    # Determine STRIDE_SIZE (can be adjusted based on experimentation)
    # A typical starting point might be 300 (same as in the original code)
    stride_size = 300
    print(f"STRIDE_SIZE: {stride_size}")

    # Plot distributions for visualization (optional)
    plt.figure(figsize=(10, 6))
    plt.hist(packet_lengths, bins=50, alpha=0.7, label='Packet Lengths')
    plt.axvline(max_packet_len, color='r', linestyle='dashed', linewidth=2, label=f'MAX_PACKET_LEN: {max_packet_len}')
    plt.hist(flow_lengths, bins=50, alpha=0.7, label='Flow Lengths')
    plt.axvline(max_flow_len, color='g', linestyle='dashed', linewidth=2, label=f'MAX_FLOW_LEN: {max_flow_len}')
    plt.legend()
    plt.xlabel('Length (Bytes)')
    plt.ylabel('Frequency')
    plt.title('Packet and Flow Length Distributions')
    plt.show()

    return max_packet_len, max_flow_len, stride_size

# Run the function to determine parameters
max_packet_len, max_flow_len, stride_size = determine_parameters()

# Use these values in your preprocessing pipeline
MAX_PACKET_LEN = max_packet_len
MAX_FLOW_LEN = max_flow_len
STRIDE_SIZE = stride_size

print(f"Using the following parameters:")
print(f"MAX_PACKET_LEN: {MAX_PACKET_LEN}")
print(f"MAX_FLOW_LEN: {MAX_FLOW_LEN}")
print(f"STRIDE_SIZE: {STRIDE_SIZE}")

# Save the determined parameters for later use
params = {
    "MAX_PACKET_LEN": MAX_PACKET_LEN,
    "MAX_FLOW_LEN": MAX_FLOW_LEN,
    "STRIDE_SIZE": STRIDE_SIZE
}

with open(os.path.join(dataset_path, "preprocessing_params.json"), 'w') as f:
    json.dump(params, f)

print("Preprocessing parameters saved!")
