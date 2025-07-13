import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# ----- STEP 1: Load and Preprocess -----
FILE_PATH = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/merged_dataset.csv"
TARGET_COLUMN = 'label'
MALICIOUS_CLIENT_NUM = 5
N_CLIENTS = 10
LABEL_FLIP_RATIO = 0.7

print("Loading dataset...")
data = pd.read_csv(FILE_PATH)
print(f"Dataset shape: {data.shape}")

# Drop non-numeric columns
non_numeric_cols = ['formatted_time', 'src_ip', 'dst_ip', 'tcp_flags', 'detailed_label']
data = data.drop(columns=non_numeric_cols)

# Encode labels
print("Encoding target labels...")
le = LabelEncoder()
data[TARGET_COLUMN] = le.fit_transform(data[TARGET_COLUMN])  # 0 or 1

# Split into train and validation
X = data.drop(columns=[TARGET_COLUMN])
y = data[TARGET_COLUMN]
X_train_full, X_val, y_train_full, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# ----- STEP 2: Split into Clients -----
print(f"Splitting data into {N_CLIENTS} clients...")
client_datasets = np.array_split(X_train_full, N_CLIENTS)
client_labels = np.array_split(y_train_full, N_CLIENTS)

# ----- STEP 3: Simulate Label Flipping Attack -----
malicious_clients = random.sample(range(N_CLIENTS), MALICIOUS_CLIENT_NUM)
print(f"Injecting label flipping into clients: {malicious_clients}")

for client_id in malicious_clients:
    labels = client_labels[client_id]
    malicious_indices = np.where(labels == 1)[0]

    if len(malicious_indices) > 0:
        n_flip = max(1, int(LABEL_FLIP_RATIO * len(malicious_indices)))
        flip_indices = np.random.choice(malicious_indices, size=n_flip, replace=False)
        labels.iloc[flip_indices] = 0
        client_labels[client_id] = labels
    else:
        print(f"Client {client_id} had no malicious samples to flip!")

# ----- STEP 4: Train Individual Client Models -----
print("\nTraining individual client models...")
client_models = []
client_preds = []

for i in range(N_CLIENTS):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
    model.fit(client_datasets[i], client_labels[i])
    preds = model.predict(X_val)
    
    client_models.append(model)
    client_preds.append(preds)

# ----- STEP 5: Calculate Client Distance Matrix -----
print("\nCalculating client distance matrix...")
client_matrix = np.hstack([pred.reshape(-1, 1) for pred in client_preds])
distance_matrix = cosine_distances(client_matrix.T)

# ----- STEP 6: Plot the Distance Matrix -----
plt.figure(figsize=(10, 8))
sns.heatmap(distance_matrix, annot=True, cmap="coolwarm",
            xticklabels=[f"C{i}" for i in range(N_CLIENTS)],
            yticklabels=[f"C{i}" for i in range(N_CLIENTS)])
plt.title("Client-to-Client Cosine Distance Matrix")
plt.show()

# ----- STEP 7: Find Outliers -----
avg_distances = np.mean(distance_matrix, axis=1)
threshold = np.median(avg_distances)
print(f"\nDistance threshold for anomaly detection: {threshold:.4f}")

suspicious_clients = [i for i, d in enumerate(avg_distances) if d > threshold]
print(f"Suspicious clients detected: {suspicious_clients}")

# ----- STEP 8: Global Performance Evaluation -----
# Majority voting using all clients
all_preds = np.array(client_preds)
majority_vote_all = np.round(np.mean(all_preds, axis=0)).astype(int)

print("\n Global performance with ALL clients (including malicious):")
print(f"Accuracy:  {accuracy_score(y_val, majority_vote_all):.4f}")
print(f"F1 Score:  {f1_score(y_val, majority_vote_all):.4f}")
print(f"Classification Report: {classification_report(y_val, majority_vote_all)}")

# Majority voting using only non-suspicious clients
benign_client_preds = [client_preds[i] for i in range(N_CLIENTS) if i not in suspicious_clients]

if len(benign_client_preds) > 0:
    benign_preds_array = np.array(benign_client_preds)
    majority_vote_cleaned = np.round(np.mean(benign_preds_array, axis=0)).astype(int)

    print("\n Global performance WITHOUT suspicious clients (defense applied):")
    print(f"Accuracy:  {accuracy_score(y_val, majority_vote_cleaned):.4f}")
    print(f"F1 Score:  {f1_score(y_val, majority_vote_cleaned):.4f}")
    print(f"Classification Report: {classification_report(y_val, majority_vote_cleaned):}")
else:
    print("\n All clients flagged as suspicious — cannot compute performance without them.")
