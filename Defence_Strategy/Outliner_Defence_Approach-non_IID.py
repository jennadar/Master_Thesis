import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.utils import shuffle
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Constants
LABEL_COLUMN = 'label'
NUM_CLIENTS = 10
DEFENSE_METHOD = 'recall'  # could be any metric
SEED = 42
POISON_FRACTION = 0.5  # Fraction of malicious clients
SOFT_FLIP_RATE = 0.5   # % of class 1 labels to flip
np.random.seed(SEED)

### Load dataset
df = pd.read_csv("D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/merged_dataset.csv")

# Encode labels
label_encoder = LabelEncoder()
df[LABEL_COLUMN] = label_encoder.fit_transform(df[LABEL_COLUMN])

# Drop non-numeric columns
object_cols = df.select_dtypes(include='object').columns
print(f"Dropping non-numeric columns: {list(object_cols)}")
df = df.drop(columns=object_cols)

### Step 1: Extreme non-IID split
def split_non_iid_extreme(df, num_clients):
    labels = df[LABEL_COLUMN].unique()
    clients_data = []
    
    for i in range(num_clients):
        major_label = labels[i % len(labels)]
        major_data = df[df[LABEL_COLUMN] == major_label]
        other_data = df[df[LABEL_COLUMN] != major_label]
        
        major_samples = major_data.sample(frac=0.8/num_clients, random_state=SEED + i)
        minor_samples = other_data.sample(frac=0.2/num_clients, random_state=SEED + i)
        
        client_df = pd.concat([major_samples, minor_samples])
        client_df = shuffle(client_df, random_state=SEED + i)
        clients_data.append(client_df.reset_index(drop=True))
    
    return clients_data

clients_data = split_non_iid_extreme(df, NUM_CLIENTS)

### Step 2: Soft Poisoning Attack
def apply_soft_poisoning(clients_data, malicious_clients):
    for i in malicious_clients:
        df = clients_data[i].copy()
        malicious_rows = df[df[LABEL_COLUMN] == 1]
        num_to_flip = int(SOFT_FLIP_RATE * len(malicious_rows))
        if num_to_flip > 0:
            indices_to_flip = malicious_rows.sample(n=num_to_flip, random_state=SEED).index
            df.loc[indices_to_flip, LABEL_COLUMN] = 0
            clients_data[i] = df
    return clients_data

malicious_clients = np.random.choice(NUM_CLIENTS, int(NUM_CLIENTS * POISON_FRACTION), replace=False)
print(f"\n Malicious Clients (Poisoned): {malicious_clients.tolist()}")

clients_data = apply_soft_poisoning(clients_data, malicious_clients)

### Step 3: Local training
client_models = []
client_metrics = []

def train_local_xgb_model(client_df):
    X = client_df.drop(columns=[LABEL_COLUMN]).values
    y = client_df[LABEL_COLUMN].values
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    preds = model.predict(X)
    return {
        'accuracy': accuracy_score(y, preds),
        'recall': recall_score(y, preds),
        'precision': precision_score(y, preds),
        'f1': f1_score(y, preds),
        'preds': preds
    }

for c_df in clients_data:
    model = train_local_xgb_model(c_df)
    client_models.append(model)
    
    X = c_df.drop(columns=[LABEL_COLUMN]).values
    y = c_df[LABEL_COLUMN].values
    metrics = evaluate_model(model, X, y)
    client_metrics.append(metrics)

### Step 4: Global ensemble via majority voting
def global_ensemble_predict(client_models, X):
    preds_all = np.array([model.predict(X) for model in client_models])
    final_preds = []
    for i in range(preds_all.shape[1]):
        vote = Counter(preds_all[:, i]).most_common(1)[0][0]
        final_preds.append(vote)
    return np.array(final_preds)

test_df = df.sample(frac=0.2, random_state=SEED)
X_test = test_df.drop(columns=[LABEL_COLUMN]).values
y_test = test_df[LABEL_COLUMN].values

global_preds = global_ensemble_predict(client_models, X_test)

def compute_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }

metrics_raw = compute_metrics(y_test, global_preds)
print("\n Global Model Metrics (Before Defense):")
print(metrics_raw)

### Step 5: Defense via recall filtering
recalls = np.array([m['recall'] for m in client_metrics])
threshold = np.median(recalls)
print(f"\n Defense Threshold (Median Recall): {threshold:.4f}")

defended_models = [model for i, model in enumerate(client_models) if client_metrics[i]['recall'] >= threshold]
defended_preds = global_ensemble_predict(defended_models, X_test)
metrics_def = compute_metrics(y_test, defended_preds)

print("\n Global Model Metrics (After Defense):")
print(metrics_def)

### Step 6: Plotting
# Before vs After Defense
labels = ['Accuracy', 'Recall', 'Precision', 'F1']
before = [metrics_raw[k.lower()] for k in labels]
after = [metrics_def[k.lower()] for k in labels]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, before, width, label='Before Defense')
plt.bar(x + width/2, after, width, label='After Defense')
plt.xticks(x, labels)
plt.ylabel('Score')
plt.title('Global Model Performance: Before vs After Defense')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Client recall plot
plt.figure(figsize=(8, 4))
plt.bar(range(NUM_CLIENTS), recalls, color='skyblue')
plt.axhline(threshold, color='red', linestyle='--', label='Defense Threshold')
plt.xlabel('Client Index')
plt.ylabel('Recall')
plt.title('Client-wise Recall')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
