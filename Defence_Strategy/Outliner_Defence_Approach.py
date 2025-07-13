import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt

# -----------------------
# Configuration
# -----------------------
N_CLIENTS = 10                 # Number of simulated clients
N_MALICIOUS = 5               # Number of poisoned clients
POISON_FLIP_RATE = 0.7         # Fraction of labels to flip in malicious clients
RANDOM_SEED = 42

# -----------------------
# Load Dataset
# -----------------------
DATA_PATH = 'D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/merged_dataset.csv'  # <<<--- Your path
TARGET_COLUMN = 'label'           # Target: 0 = benign, 1 = malicious

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded. Shape: {df.shape}")

label_mapping = {'benign': 0, 'malicious': 1}
df[TARGET_COLUMN] = df[TARGET_COLUMN].map(label_mapping)

# Drop non-numeric/object columns
object_cols = df.select_dtypes(include='object').columns
print(f"Dropping non-numeric columns: {list(object_cols)}")
df = df.drop(columns=object_cols)

# Validation split
validation_df = df.sample(frac=0.1, random_state=RANDOM_SEED)
df = df.drop(validation_df.index)

print(f"Validation set created. Shape: {validation_df.shape}")

# Train-validation split from remaining data
train_df, validation_df = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
X_val = validation_df.drop(columns=[TARGET_COLUMN])
y_val = validation_df[TARGET_COLUMN]

# -----------------------
# Split into Clients
# -----------------------
clients_data = np.array_split(train_df, N_CLIENTS)
print(f"Data split into {N_CLIENTS} clients.")

# -----------------------
# Select malicious clients
# -----------------------
malicious_clients = random.sample(range(N_CLIENTS), N_MALICIOUS)
print(f"Malicious clients are: {malicious_clients}")

# -----------------------
# Poison function
# -----------------------
def poison_labels(df, flip_rate=0.3):
    poisoned_df = df.copy()
    malicious_samples = poisoned_df[poisoned_df[TARGET_COLUMN] == 1]
    num_to_flip = int(flip_rate * len(malicious_samples))
    if num_to_flip == 0:
        return poisoned_df
    flip_indices = malicious_samples.sample(n=num_to_flip, random_state=RANDOM_SEED).index
    poisoned_df.loc[flip_indices, TARGET_COLUMN] = 0
    return poisoned_df

# -----------------------
# Train Client Models
# -----------------------
client_models = []
client_recalls = []
client_precisions = []

for client_id, client_df in enumerate(clients_data):
    print(f"\nTraining Client {client_id}...")

    if client_id in malicious_clients:
        client_df = poison_labels(client_df, flip_rate=POISON_FLIP_RATE)
        print(f"Client {client_id} data poisoned.")

    X_train = client_df.drop(columns=[TARGET_COLUMN])
    y_train = client_df[TARGET_COLUMN]

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    recall = recall_score(y_val, y_pred, zero_division=0)
    precision = precision_score(y_val, y_pred, zero_division=0)

    print(f"Recall on validation: {recall:.4f}, Precision: {precision:.4f}")

    client_models.append(model)
    client_recalls.append(recall)
    client_precisions.append(precision)

# -----------------------
# Outlier Detection using Median Recall
# -----------------------
OUTLIER_RECALL_THRESHOLD = np.median(client_recalls)
print(f"\nUsing median recall as threshold: {OUTLIER_RECALL_THRESHOLD:.4f}")

good_clients = [i for i, rec in enumerate(client_recalls) if rec >= OUTLIER_RECALL_THRESHOLD]
bad_clients = [i for i, rec in enumerate(client_recalls) if rec < OUTLIER_RECALL_THRESHOLD]

print("Good Clients (used for aggregation):", good_clients)
print("Detected Bad Clients (excluded):", bad_clients)

# -----------------------
# Aggregation Function
# -----------------------
def aggregate_predictions(models, X):
    preds = np.zeros((len(models), X.shape[0]))
    for idx, model in enumerate(models):
        preds[idx] = model.predict(X)
    return np.round(np.mean(preds, axis=0))

# Global prediction without defense
global_preds_no_defense = aggregate_predictions(client_models, X_val)

# Global prediction with defense
good_models = [client_models[i] for i in good_clients]
global_preds_with_defense = aggregate_predictions(good_models, X_val)

# -----------------------
# Evaluation Function
# -----------------------
def evaluate_global(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true,y_pred, zero_division=0)
    return acc, rec, prec, f1

acc_no_def, rec_no_def, prec_no_def, f1_no_def = evaluate_global(y_val, global_preds_no_defense)
acc_def, rec_def, prec_def, f1_def= evaluate_global(y_val, global_preds_with_defense)

# -----------------------
# Print Global Evaluation
# -----------------------
print("\n--- Global Model Evaluation ---")
print(f"Without Defense --> Accuracy: {acc_no_def:.4f}, Recall: {rec_no_def:.4f}, Precision: {prec_no_def:.4f}, f1_score: {f1_no_def}")
print(f"With Defense    --> Accuracy: {acc_def:.4f}, Recall: {rec_def:.4f}, Precision: {prec_def:.4f}, f1_score: {f1_def}")

# -----------------------
# Plot: Client Recalls
# -----------------------
plt.figure(figsize=(10, 6))
bar_colors = ['red' if i in bad_clients else 'green' for i in range(N_CLIENTS)]
plt.bar(range(N_CLIENTS), client_recalls, color=bar_colors)
plt.axhline(y=OUTLIER_RECALL_THRESHOLD, color='black', linestyle='--', label='Median Threshold')
plt.xlabel('Client ID')
plt.ylabel('Recall on Validation Set')
plt.title('Client Recalls (Green = Good, Red = Bad)')
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------
# Plot: Global Model Comparison
# -----------------------

metrics = ['Accuracy', 'Recall', 'F1_score']
before_def = [acc_no_def, rec_no_def, f1_no_def]
after_def = [acc_def, rec_def, f1_def]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, before_def, width, label='Without Defense', color='tomato')
plt.bar(x + width/2, after_def, width, label='With Defense', color='seagreen')
plt.xticks(x, metrics)
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Global Model Performance Comparison')
plt.legend()
plt.tight_layout()
plt.show()



""" import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt

# -----------------------
# Configuration
# -----------------------
N_CLIENTS = 10                 # Number of simulated clients
N_MALICIOUS = 2                # Number of poisoned clients
POISON_FLIP_RATE = 0.5         # Fraction of labels to flip in malicious clients
OUTLIER_RECALL_THRESHOLD = 0.5 # Recall threshold to detect malicious clients
RANDOM_SEED = 42

# -----------------------
# Load Dataset
# -----------------------
DATA_PATH = 'D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/merged_dataset.csv'  # <<<--- CHANGE THIS to your feature.csv path
TARGET_COLUMN = 'label'           # Assuming 'label' is your target (0: benign, 1: malicious)

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded. Shape: {df.shape}")

label_mapping = {'benign': 0, 'malicious': 1}
df[TARGET_COLUMN] = df[TARGET_COLUMN].map(label_mapping)

# Drop object columns that are not useful for training
object_cols = df.select_dtypes(include='object').columns
print(f"Dropping non-numeric columns: {list(object_cols)}")
df = df.drop(columns=object_cols)

# Optional: Shuffle the dataset
#df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
validation_df = df.sample(frac=0.1, random_state=RANDOM_SEED)
df = df.drop(validation_df.index)

# Also map validation set labels
validation_df[TARGET_COLUMN] = validation_df[TARGET_COLUMN].map(label_mapping)

print(f"Validation set created. Shape: {validation_df.shape}")
# -----------------------
# Create clean validation set (for client evaluation)
# -----------------------
train_df, validation_df = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
print(f"Validation set created. Shape: {validation_df.shape}")

X_val = validation_df.drop(columns=[TARGET_COLUMN])
y_val = validation_df[TARGET_COLUMN]

# -----------------------
# Split into Clients
# -----------------------
clients_data = np.array_split(train_df, N_CLIENTS)
print(f"Data split into {N_CLIENTS} clients.")

# -----------------------
# Select malicious clients
# -----------------------
malicious_clients = random.sample(range(N_CLIENTS), N_MALICIOUS)
print(f"Malicious clients are: {malicious_clients}")

# -----------------------
# Poison function
# -----------------------
def poison_labels(df, flip_rate=0.3):
    poisoned_df = df.copy()
    # Select only malicious samples
    malicious_samples = poisoned_df[poisoned_df[TARGET_COLUMN] == 1]
    
    num_to_flip = int(flip_rate * len(malicious_samples))
    if num_to_flip == 0:
        return poisoned_df  # nothing to flip

    # Randomly select malicious samples to flip
    flip_indices = malicious_samples.sample(n=num_to_flip, random_state=RANDOM_SEED).index
    poisoned_df.loc[flip_indices, TARGET_COLUMN] = 0  # Flip to benign (0)

    return poisoned_df
    
# -----------------------
# Train Client Models
# -----------------------
client_models = []
client_recalls = []
client_precisions = []

for client_id, client_df in enumerate(clients_data):
    print(f"\nTraining Client {client_id}...")

    if client_id in malicious_clients:
        client_df = poison_labels(client_df, flip_rate=POISON_FLIP_RATE)
        print(f"Client {client_id} data poisoned.")

    X_train = client_df.drop(columns=[TARGET_COLUMN])
    y_train = client_df[TARGET_COLUMN]

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    recall = recall_score(y_val, y_pred, zero_division=0)
    precision = precision_score(y_val, y_pred, zero_division=0)

    print(f"Recall on validation: {recall:.4f}, Precision: {precision:.4f}")

    client_models.append(model)
    client_recalls.append(recall)
    client_precisions.append(precision)

# -----------------------
# Outlier Detection
# -----------------------
good_clients = [i for i, rec in enumerate(client_recalls) if rec >= OUTLIER_RECALL_THRESHOLD]
bad_clients = [i for i, rec in enumerate(client_recalls) if rec < OUTLIER_RECALL_THRESHOLD]

print("\nGood Clients (used for aggregation):", good_clients)
print("Detected Bad Clients (excluded):", bad_clients)

# -----------------------
# Aggregation of Good Clients
# -----------------------

# Simple aggregation via soft-voting on validation set
def aggregate_predictions(models, X):
    preds = np.zeros((len(models), X.shape[0]))
    for idx, model in enumerate(models):
        preds[idx] = model.predict(X)
    # Majority vote
    final_preds = np.round(np.mean(preds, axis=0))
    return final_preds

# Aggregation without defense (all clients, poisoned included)
global_preds_no_defense = aggregate_predictions(client_models, X_val)

# Aggregation with defense (only good clients)
good_models = [client_models[i] for i in good_clients]
global_preds_with_defense = aggregate_predictions(good_models, X_val)

# -----------------------
# Evaluation
# -----------------------
print("\n--- Global Model Evaluation ---")

def evaluate_global(y_true, y_pred, title):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    print(f"{title} --> Accuracy: {acc:.4f}, Recall: {rec:.4f}, Precision: {prec:.4f}")

evaluate_global(y_val, global_preds_no_defense, "Without Defense")
evaluate_global(y_val, global_preds_with_defense, "With Defense (Outlier Removal)")

# -----------------------
# Plot Client Recalls
# -----------------------
plt.figure(figsize=(10,6))
plt.bar(range(N_CLIENTS), client_recalls, color=['red' if i in bad_clients else 'green' for i in range(N_CLIENTS)])
plt.axhline(y=OUTLIER_RECALL_THRESHOLD, color='black', linestyle='--')
plt.xlabel('Client ID')
plt.ylabel('Recall on Validation Set')
plt.title('Client Recalls (Green=Good, Red=Bad)')
plt.show()
 """