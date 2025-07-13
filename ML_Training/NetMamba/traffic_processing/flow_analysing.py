'''
import os
import json
import numpy as np

# Directory containing flows
flow_dir = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/CTU-IoT-Malware-Capture-21-1/flow_data"

# Lists to store flow statistics
packet_lengths = []
flow_lengths = []

# Process each flow file
for flow_file in os.listdir(flow_dir):
    with open(os.path.join(flow_dir, flow_file), "r") as f:
        flow_data = json.load(f)
    
    # Extract packet lengths
    packet_lengths.extend([pkt["packet_length"] for pkt in flow_data])
    
    # Store flow length (number of packets in flow)
    flow_lengths.append(len(flow_data))

# Convert to NumPy arrays
packet_lengths = np.array(packet_lengths)
flow_lengths = np.array(flow_lengths)

# Compute statistics
print("Packet Length Stats:")
print(f"  Min: {packet_lengths.min()}, Max: {packet_lengths.max()}, Mean: {packet_lengths.mean()}, Median: {np.median(packet_lengths)}")

print("\nFlow Length Stats:")
print(f"  Min: {flow_lengths.min()}, Max: {flow_lengths.max()}, Mean: {flow_lengths.mean()}, Median: {np.median(flow_lengths)}")

# Save the computed stats (optional)
np.save("packet_lengths.npy", packet_lengths)
np.save("flow_lengths.npy", flow_lengths)
'''

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Directory containing flows
flow_dir = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/CTU-IoT-Malware-Capture-1-1/flow_data"

# Lists to store flow statistics
packet_lengths = []
flow_lengths = []

# Process each flow file
for flow_file in os.listdir(flow_dir):
    with open(os.path.join(flow_dir, flow_file), "r") as f:
        flow_data = json.load(f)

    # Extract packet lengths
    packet_lengths.extend([pkt["packet_length"] for pkt in flow_data])

    # Store flow length (number of packets in flow)
    flow_lengths.append(len(flow_data))

# Convert to NumPy arrays
packet_lengths = np.array(packet_lengths)
flow_lengths = np.array(flow_lengths)

# Compute statistics
print("Packet Length Stats:")
print(f"  Min: {packet_lengths.min()}, Max: {packet_lengths.max()}, Mean: {packet_lengths.mean()}, Median: {np.median(packet_lengths)}")

print("\nFlow Length Stats:")
print(f"  Min: {flow_lengths.min()}, Max: {flow_lengths.max()}, Mean: {flow_lengths.mean()}, Median: {np.median(flow_lengths)}")

# Save the computed stats (optional)
#np.save("packet_lengths.npy", packet_lengths)
#np.save("flow_lengths.npy", flow_lengths)

# Plot 1: Histogram of Packet Sizes
plt.figure(figsize=(10, 5))
plt.hist(packet_lengths, bins=50, color="blue", alpha=0.7, edgecolor="black")
plt.xlabel("Packet Size (bytes)")
plt.ylabel("Frequency")
plt.title("Distribution of Packet Sizes")
plt.yscale("log")  # Set y-axis to logarithmic scale
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Plot 2: Histogram of Flow Lengths
plt.figure(figsize=(10, 5))
plt.hist(flow_lengths, bins=50, color="red", alpha=0.7, edgecolor="black")
plt.xlabel("Number of Packets per Flow")
plt.ylabel("Frequency")
plt.title("Distribution of Flow Lengths")
plt.yscale("log")  # Set y-axis to logarithmic scale
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
