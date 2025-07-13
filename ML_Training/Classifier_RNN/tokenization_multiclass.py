import pandas as pd
import numpy as np
import os
import json
from collections import defaultdict, Counter
from sklearn.preprocessing import KBinsDiscretizer
from tqdm import tqdm

# ====================== Configuration ======================
BIN_COUNT = 5        # Number of bins for packet_length
MAX_SEQ_LEN = 100    # Max tokens per flow (pad or truncate)
VOCAB_SAVE_PATH = 'Token_Labels/Merged_dataset/vocab_mapping.json'
SEQUENCE_SAVE_PATH = 'Token_Labels/Merged_dataset/tokenized_sequences.npy'
LABEL_SAVE_PATH = 'Token_Labels/Merged_dataset/labels.npy'
CLASS_MAPPING_SAVE_PATH = 'Token_Labels/Merged_dataset/class_mapping.json'

# ====================== Load CSV ===========================
df = pd.read_csv("D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/merged_dataset.csv")

# Ensure proper timestamp sorting
df.sort_values(by='ts', inplace=True)

# Fill missing TCP flags
df['tcp_flags'] = df['tcp_flags'].fillna("NONE")

# Replace '-' in detailed_label with 'Benign'
df['detailed_label'] = df['detailed_label'].replace('-', 'Benign')

# Optional: Convert packet_length to numeric
df['packet_length'] = pd.to_numeric(df['packet_length'], errors='coerce').fillna(0)

# ====================== Step 1: Bin packet_length ======================
def bin_packet_size(length):
    if length < 100:
        return 'LEN_SMALL'
    elif length < 500:
        return 'LEN_MED'
    elif length < 1000:
        return 'LEN_LARGE'
    else:
        return 'LEN_XL'

# ====================== Step 2: Construct token ======================
def construct_token(row):
    proto = str(row['proto']).upper()
    dst_port = str(row['dst_port'])
    pkt_bin = bin_packet_size(row['packet_length'])
    flags = str(row['tcp_flags']).upper()
    token = f"{proto}_{dst_port}_{pkt_bin}_{flags}"
    return token

df['token'] = df.apply(construct_token, axis=1)

# ====================== Step 3: Define Flow ID ======================
df['flow_id'] = df['src_ip'].astype(str) + "_" + df['dst_ip'].astype(str) + "_" + df['proto'].astype(str) + "_" + df['dst_port'].astype(str)

# ====================== Step 4: Build Vocabulary and Sequences ======================
token_counts = defaultdict(int)
all_flows = {}
flow_labels = {}

# We'll store all unique classes
all_classes = set()

for flow_id, group in tqdm(df.groupby('flow_id')):
    tokens = group['token'].tolist()
    detailed_labels = group['detailed_label'].tolist()
    
    # Count attack types
    label_counter = Counter(detailed_labels)
    
    # Pick most frequent attack type in this flow
    flow_label = label_counter.most_common(1)[0][0]
    
    # Update class set
    all_classes.add(flow_label)

    for t in tokens:
        token_counts[t] += 1
    all_flows[flow_id] = tokens[:MAX_SEQ_LEN]  # truncate if too long
    flow_labels[flow_id] = flow_label

# ====================== Step 5: Create Token Vocabulary ======================
token_to_idx = {'PAD': 0, 'UNK': 1}
for token in token_counts:
    if token not in token_to_idx:
        token_to_idx[token] = len(token_to_idx)

# ====================== Step 6: Encode Class Labels ======================
class_list = sorted(list(all_classes))
class_to_idx = {label: idx for idx, label in enumerate(class_list)}

# Save class mapping
with open(CLASS_MAPPING_SAVE_PATH, 'w') as f:
    json.dump(class_to_idx, f, indent=2)

# ====================== Step 7: Convert to Sequences ======================
tokenized_sequences = []
label_list = []

for flow_id, tokens in all_flows.items():
    indexed = [token_to_idx.get(t, token_to_idx['UNK']) for t in tokens]
    # Pad sequence
    if len(indexed) < MAX_SEQ_LEN:
        indexed += [token_to_idx['PAD']] * (MAX_SEQ_LEN - len(indexed))
    else:
        indexed = indexed[:MAX_SEQ_LEN]
    
    tokenized_sequences.append(indexed)
    
    label_str = flow_labels[flow_id]
    label_list.append(class_to_idx[label_str])

tokenized_sequences = np.array(tokenized_sequences)
label_list = np.array(label_list)

# ====================== Save Everything ======================
np.save(SEQUENCE_SAVE_PATH, tokenized_sequences)
np.save(LABEL_SAVE_PATH, label_list)

with open(VOCAB_SAVE_PATH, 'w') as f:
    json.dump(token_to_idx, f, indent=2)

print("Tokenization complete!")
print(f"Saved {len(token_to_idx)} vocabulary tokens.")
print(f"Saved {len(class_to_idx)} class labels.")
print(f"Saved {tokenized_sequences.shape[0]} tokenized sequences to '{SEQUENCE_SAVE_PATH}'")
