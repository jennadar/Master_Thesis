import pandas as pd
import numpy as np
from collections import Counter
import json
import os

# ========== CONFIG ==========
MAX_SEQ_LEN = 10
VOCAB_OUTPUT = 'Token_Labels/Binary/vocab_mapping.json'
PROCESSED_OUTPUT_X = 'Token_Labels/Binary/tokenized_sequences1.npy'
PROCESSED_OUTPUT_Y = "Token_Labels/Binary/labels.npy"
CSV_INPUT = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/Train_dataset/2018-05-09-192.168.100.103.horizontalPort_packet_flow_features_merged_labeled.csv"  # change if using another CSV

# ========== STEP 1: LOAD DATA ==========
print("🔹 Loading CSV...")
df = pd.read_csv(CSV_INPUT)

# ========== STEP 2: BASIC CLEANING ==========
#df = df.dropna(subset=['proto', 'src_port', 'dst_port', 'packet_length', 'tcp_flags', 'label'])

# ========== STEP 3: CREATE TOKENS FROM FLOW ==========
def create_tokens_from_flow(row):
    tokens = []
    # Protocol
    tokens.append(f"proto:{row['proto']}")
    
    # Source and destination port bucketing
    tokens.append(f"src_port:{int(row['src_port']) // 1000 * 1000}")
    tokens.append(f"dst_port:{int(row['dst_port']) // 1000 * 1000}")
    
    # Packet length binning
    pkt_len = int(row['packet_length'])
    if pkt_len < 100:
        tokens.append("pkt_len:<100")
    elif pkt_len < 500:
        tokens.append("pkt_len:100-500")
    else:
        tokens.append("pkt_len:>500")
    
    # TCP Flags
    tokens.append(f"flags:{row['tcp_flags']}")
    
    return tokens

# ========== STEP 4: BUILD VOCAB ==========
print("🔹 Building vocabulary...")
token_counter = Counter()
for _, row in df.iterrows():
    tokens = create_tokens_from_flow(row)
    token_counter.update(tokens)

# Special tokens
vocab = ['<PAD>', '<UNK>'] + [tok for tok, _ in token_counter.most_common()]
token2idx = {tok: idx for idx, tok in enumerate(vocab)}

# Save vocab
with open(VOCAB_OUTPUT, "w") as f:
    json.dump(token2idx, f)
print(f"🔸 Saved token vocabulary to: {VOCAB_OUTPUT}")

# ========== STEP 5: TOKENIZATION + PADDING ==========
def tokenize_and_pad(tokens, token2idx, max_len=MAX_SEQ_LEN):
    indices = [token2idx.get(tok, token2idx['<UNK>']) for tok in tokens]
    if len(indices) < max_len:
        indices += [token2idx['<PAD>']] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices

X = []
y = []

print("🔹 Tokenizing sequences...")
for _, row in df.iterrows():
    tokens = create_tokens_from_flow(row)
    sequence = tokenize_and_pad(tokens, token2idx)
    X.append(sequence)
    
    # Binary Label: 1 = malicious, 0 = benign
    label_str = str(row['label']).lower()
    label = 1 if "malicious" in label_str else 0
    y.append(label)

X = np.array(X)
y = np.array(y)

# ========== STEP 6: SAVE ==========
np.save(PROCESSED_OUTPUT_X, X)
np.save(PROCESSED_OUTPUT_Y, y)
print(f"✅ Saved sequences to {PROCESSED_OUTPUT_X} and labels to {PROCESSED_OUTPUT_Y}")
print(f"🧾 Total samples: {len(X)}, Positive (malicious): {np.sum(y)}, Negative (benign): {len(y) - np.sum(y)}")
