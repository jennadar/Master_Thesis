#2 step
import pandas as pd

def parse_conn_log_labeled(log_file):
    """ Parse Zeek conn.log.labeled file and extract relevant fields including detailed-label """
    labels = []
    with open(log_file, "r") as f:
        for line in f:
            if line.startswith("#"):  # Skip Zeek headers
                continue
            fields = line.strip().split()
            if len(fields) < 9:  # Ensure there are enough fields for label + detailed-label
                continue

            ts = round(float(fields[0]), 6)  # Convert timestamp (trim extra precision)
            src_ip = fields[2]  # id.orig_h (source IP)
            label = fields[-2]   # The second last column is the general label (Benign/Malicious)
            detailed_label = fields[-1]  # The last column is the detailed attack/benign label

            labels.append((ts, src_ip, label, detailed_label))
    return labels

def map_labels_to_features(feature_csv, log_file, output_csv):
    """ Map labels and detailed labels from conn.log.labeled to extracted features CSV """
    df = pd.read_csv(feature_csv)
    labeled_data = parse_conn_log_labeled(log_file)

    # Convert timestamps in df to match precision of conn.log.labeled
    df["ts"] = df["ts"].round(6)

    # Create a dictionary for quick lookup
    label_dict = {(ts, src_ip): (label, detailed_label) for ts, src_ip, label, detailed_label in labeled_data}

    # Assign labels based on matching (ts, src_ip)
    df["label"] = df.apply(lambda row: label_dict.get((row["ts"], row["src_ip"]), ("Unknown", "Unknown"))[0], axis=1)
    df["detailed_label"] = df.apply(lambda row: label_dict.get((row["ts"], row["src_ip"]), ("Unknown", "Unknown"))[1], axis=1)

    # Handle Unknown Labels
    df = classify_unknown_labels(df)
    return df

def classify_unknown_labels(df):
    """ Classifies unknown labels & detailed_labels using flow_duration and previous packet labels """
    for i in range(len(df)):
        if df.loc[i, "label"] == "Unknown":
            flow_duration = df.loc[i, "flow_duration"]
            
            if flow_duration > 1:
                df.loc[i, "label"] = "Benign"
                df.loc[i, "detailed_label"] = "-"
            else:
                # Find last known label and detailed_label in the same flow
                src_ip, dst_ip, proto, src_port, dst_port = df.loc[i, ["src_ip", "dst_ip", "proto", "src_port", "dst_port"]]
                flow_packets = df[
                    (df["src_ip"] == src_ip) & 
                    (df["dst_ip"] == dst_ip) & 
                    (df["proto"] == proto) & 
                    (df["src_port"] == src_port) & 
                    (df["dst_port"] == dst_port)
                ]
                
                # Get the last known label & detailed_label in the flow
                known_labels = flow_packets[flow_packets["label"] != "Unknown"]
                if not known_labels.empty:
                    last_known_label = known_labels["label"].iloc[-1]
                    last_known_detailed_label = known_labels["detailed_label"].iloc[-1]
                else:
                    last_known_label = "Benign"
                    last_known_detailed_label = "-"

                df.loc[i, "label"] = last_known_label
                df.loc[i, "detailed_label"] = last_known_detailed_label

    return df

# Example Usage
feature_csv = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/CTU-IoT-Malware-Capture-21-1/2018-10-03-15-22-32-192.168.100.113_packet_flow_features.csv"
#feature_csv = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/CTU-Honeypot-Capture-4-1/2018-09-14-13-40-25-Philips-Hue-Bridge_packet_flow_features.csv"
conn_log_labeled = "D:/jenny/Documents/FAUS_Study/Thesis/iot_23_datasets_full/IoTScenarios/CTU-IoT-Malware-Capture-21-1/bro/conn.log.labeled"  # Replace with actual path
output_csv = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/CTU-IoT-Malware-Capture-21-1/2018-10-03-15-22-32-192.168.100.113_packet_flow_features_labeled.csv"

df = map_labels_to_features(feature_csv, conn_log_labeled, output_csv)
df.to_csv(output_csv, index=False)

print(f"Final labeled data saved to {output_csv}")