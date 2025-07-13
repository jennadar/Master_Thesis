import os
import json
import numpy as np
import pandas as pd
from collections import Counter

# === PATHS ===
flow_dir = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/CTU-IoT-Malware-Capture-1-1/flow_data"
stride_output_dir = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/CTU-IoT-Malware-Capture-1-1/strides_data"
label_output_file = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/CTU-IoT-Malware-Capture-1-1/strides_data1/stride_labels.csv"

# === PARAMETERS ===
MAX_PACKET_LENGTH = 512  # Crop packets to this size (bytes)
MAX_FLOW_LEN = 20        # Max number of packets per flow
STRIDE_SIZE = 512        # Number of bytes per stride
MALICIOUS_THRESHOLD = 1  # Min malicious packets per stride for labeling
PADDING_BYTE = b'\x00'   # Padding value (zero bytes)

# Store stride labels
stride_labels = []

# Ensure output directories exist
os.makedirs(stride_output_dir, exist_ok=True)

# Process each flow file
for flow_file in os.listdir(flow_dir):
    with open(os.path.join(flow_dir, flow_file), "r") as f:
        flow_data = json.load(f)

    if len(flow_data) == 0:
        continue  # Skip empty flows

    # === 1. PACKET PARSING & CROPPING ===
    packet_byte_arrays = []
    labels = []
    detailed_labels = []

    for pkt in flow_data:
        pkt_length = min(pkt["packet_length"], MAX_PACKET_LENGTH)  # Crop packet if too long
        packet_byte_array = bytes(pkt_length)  # Simulated byte extraction

        # Pad shorter packets
        if pkt_length < MAX_PACKET_LENGTH:
            packet_byte_array += PADDING_BYTE * (MAX_PACKET_LENGTH - pkt_length)

        packet_byte_arrays.append(packet_byte_array)
        labels.append(pkt["label"])
        detailed_labels.append(pkt["detailed_label"])

    # === 2. FLOW PADDING ===
    if len(packet_byte_arrays) < MAX_FLOW_LEN:
        num_padding_packets = MAX_FLOW_LEN - len(packet_byte_arrays)
        packet_byte_arrays.extend([PADDING_BYTE * MAX_PACKET_LENGTH] * num_padding_packets)
        labels.extend(["Benign"] * num_padding_packets)  # Default benign label
        detailed_labels.extend(["-"] * num_padding_packets)

    # Truncate flow if too long
    packet_byte_arrays = packet_byte_arrays[:MAX_FLOW_LEN]
    labels = labels[:MAX_FLOW_LEN]
    detailed_labels = detailed_labels[:MAX_FLOW_LEN]

    # === 3. PACKET CONCATENATION INTO BYTE ARRAY ===
    flow_byte_array = b"".join(packet_byte_arrays)

    # === 4. STRIDE CUTTING ===
    num_strides = len(flow_byte_array) // STRIDE_SIZE
    strides = [flow_byte_array[i * STRIDE_SIZE:(i + 1) * STRIDE_SIZE] for i in range(num_strides)]

    for i, stride in enumerate(strides):
        # Get labels for this stride
        start_idx = i * (len(labels) // num_strides)
        end_idx = (i + 1) * (len(labels) // num_strides)
        stride_labels_list = labels[start_idx:end_idx]
        stride_detailed_labels = detailed_labels[start_idx:end_idx]

        # === 5. STRIDE LABELING ===
        malicious_count = sum(1 for label in stride_labels_list if label != "Benign")
        if malicious_count >= MALICIOUS_THRESHOLD:
            stride_label = "Malicious"
            stride_attack_type = Counter([lbl for lbl in stride_detailed_labels if lbl != "-"]).most_common(1)[0][0]
        else:
            stride_label = "Benign"
            stride_attack_type = "-"

        # Pad stride if shorter than STRIDE_SIZE
        if len(stride) < STRIDE_SIZE:
            stride += PADDING_BYTE * (STRIDE_SIZE - len(stride))

        # Save stride as .npy
        stride_filename = f"{flow_file}_stride_{i}.npy"
        np.save(os.path.join(stride_output_dir, stride_filename), np.frombuffer(stride, dtype=np.uint8))

        # Save label
        stride_labels.append([stride_filename, stride_label, stride_attack_type])

# === Save Labels to CSV ===
df = pd.DataFrame(stride_labels, columns=["file", "label", "detailed_label"])
df.to_csv(label_output_file, index=False)

print("Stride processing completed with NetMamba-compatible formatting!")
