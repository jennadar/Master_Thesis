import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# -------------------- Paths ---------------------
#train_file = r"D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/Train_dataset/2018-05-09-192.168.100.103.horizontalPort_packet_flow_features_merged_labeled.csv"
train_file = r"D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/Train_dataset/2L-feature_mislabelled_dataset_20.csv"
test_file = r"D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/Train_dataset/test_dataset.csv"
model_save_path = r"D:/jenny/Documents/FAUS_Study/Thesis/RNN_Transformer/Models/Standard_RNN_binary_classifier/"

# -------------------- Feature Selection ---------------------
selected_features = [
    'packet_length', 'raw_len', 'inter_packet_time', 'flow_duration',
    'total_packets_in_flow', 'total_bytes_in_flow', 'mean_packet_size',
    'mean_inter_arrival_time', 'packets_per_second', 'bytes_per_second'
]

# -------------------- Helper Functions ---------------------
def preprocess(df):
    # Binary Label
    df['label_binary'] = df['label'].apply(lambda x: 1 if x.lower() == 'malicious' else 0)

    # Drop rows with missing selected features
    df = df.dropna(subset=selected_features)

    # Normalize
    scaler = StandardScaler()
    df[selected_features] = scaler.fit_transform(df[selected_features])

    return df, scaler

def create_sequences(df, max_len=20):
    flows = df.groupby(['src_ip', 'dst_ip', 'src_port', 'dst_port', 'proto'])
    sequences, labels = [], []

    for _, group in flows:
        features_tensor = torch.tensor(group[selected_features].values, dtype=torch.float32)
        label = group['label_binary'].iloc[0]
        
        # Truncate or pad
        if len(features_tensor) > max_len:
            features_tensor = features_tensor[:max_len]
        else:
            pad_size = max_len - len(features_tensor)
            padding = torch.zeros((pad_size, len(selected_features)))
            features_tensor = torch.cat((features_tensor, padding), dim=0)
        
        sequences.append(features_tensor)
        labels.append(label)

    return torch.stack(sequences), torch.tensor(labels, dtype=torch.float32)

# -------------------- Model ---------------------
class TrafficRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super(TrafficRNN, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.sigmoid(self.fc(out))

# -------------------- Training ---------------------
def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, device='cpu'):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    best_model_state = None
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        train_losses.append(total_train_loss / len(train_loader))

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
                pred = model(xb)
                loss = criterion(pred, yb)
                total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict()

    print("Training complete. Best validation loss:", best_loss)
    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses

# -------------------- Evaluation ---------------------
""" def evaluate(model, X, y, device='cpu'):
    model.eval()
    with torch.no_grad():
        preds = model(X.to(device)).cpu().numpy()
    preds_binary = (preds > 0.5).astype(int)
    print(classification_report(y, preds_binary, target_names=["Benign", "Malicious"]))

    # Confusion Matrix
    cm = confusion_matrix(y, preds_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malicious"])
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    # Prediction Plot
    plt.figure(figsize=(10, 4))
    plt.plot(preds[:100], label='Predicted Probabilities')
    plt.plot(y[:100].numpy(), label='True Labels', alpha=0.7)
    plt.legend()
    plt.title("Prediction Probabilities vs True Labels (First 100 Samples)")
    plt.xlabel("Sample")
    plt.ylabel("Probability / Label")
    plt.grid(True)
    plt.show()
 """
def evaluate(model, X, y, device='cpu', batch_size=64):
    model.eval()
    dataloader = DataLoader(TensorDataset(X, y), batch_size=batch_size)
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(yb.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    preds_binary = (all_preds > 0.5).astype(int)

    print(classification_report(all_labels, preds_binary, target_names=["Benign", "Malicious"]))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, preds_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malicious"])
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    # Prediction Plot
    plt.figure(figsize=(10, 4))
    plt.plot(all_preds[:100], label='Predicted Probabilities')
    plt.plot(all_labels[:100], label='True Labels', alpha=0.7)
    plt.legend()
    plt.title("Prediction Probabilities vs True Labels (First 100 Samples)")
    plt.xlabel("Sample")
    plt.ylabel("Probability / Label")
    plt.grid(True)
    plt.show()

# -------------------- Run Pipeline ---------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess training data
df_train = pd.read_csv(train_file)
df_train, scaler = preprocess(df_train)
X_seq, y_seq = create_sequences(df_train)

# Train/Validation Split
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=42)

# DataLoaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

# Model
model = TrafficRNN(input_dim=len(selected_features)).to(device)

# Train
model, train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=20, device=device)

# Save the model
os.makedirs(model_save_path, exist_ok=True)
torch.save(model.state_dict(), os.path.join(model_save_path, "rnn_mislabeled_binary_classifier-20.pth"))

# Evaluate on validation set
print("\n--- Evaluation on Validation Set ---")
evaluate(model, X_val, y_val, device=device)

# -------------------- Test on Unseen Data ---------------------
print("\n--- Evaluation on Unseen Test Dataset ---")
df_test = pd.read_csv(test_file)
df_test, _ = preprocess(df_test)  # Use same scaler if needed
X_test_seq, y_test_seq = create_sequences(df_test)
evaluate(model, X_test_seq, y_test_seq, device=device)
