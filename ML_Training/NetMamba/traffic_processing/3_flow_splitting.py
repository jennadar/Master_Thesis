import os
import pandas as pd
import json
from collections import defaultdict

# Load the dataset
df = pd.read_csv("D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/CTU-IoT-Malware-Capture-1-1/2018-05-09-192.168.100.103.horizontalPort_packet_flow_features_merged_labeled.csv")

# Group by flows (identified by these fields)
flow_groups = df.groupby(["src_ip", "dst_ip", "src_port", "dst_port", "proto"])

# Create directory to store flows
output_dir = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/CTU-IoT-Malware-Capture-1-1/flow_data"
os.makedirs(output_dir, exist_ok=True)

# Process each flow
flow_dict = defaultdict(list)

for (src_ip, dst_ip, src_port, dst_port, proto), flow in flow_groups:
    flow_data = []
    
    for _, row in flow.iterrows():
        packet_info = {
            "packet_length": row["packet_length"],
            "inter_packet_time": row["inter_packet_time"],
            "tcp_flags": row["tcp_flags"],
            "flow_duration": row["flow_duration"],
            "total_bytes_in_flow": row["total_bytes_in_flow"],
            "mean_packet_size": row["mean_packet_size"],
            "mean_inter_arrival_time": row["mean_inter_arrival_time"],
            "packets_per_second": row["packets_per_second"],
            "bytes_per_second": row["bytes_per_second"],
            "label": row["label"],  # Store label for later
            "detailed_label": row["detailed_label"]
        }
        flow_data.append(packet_info)

    # Save the flow as JSON
    flow_id = f"{src_ip}_{dst_ip}_{src_port}_{dst_port}_{proto}"
    with open(os.path.join(output_dir, f"{flow_id}.json"), "w") as f:
        json.dump(flow_data, f)

print("Flows saved successfully!")
