import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import joblib

# Assuming your RNN model is already defined, and you're using a dataset
# This code focuses on the training process and memory management adjustments

# Model class definition (simplified for this example)
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Get the output of the last time step
        out = self.fc(out)
        return out


# Load and process data
df = pd.read_csv("D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/Train_dataset/2018-05-09-192.168.100.103.horizontalPort_packet_flow_features_merged_labeled.csv")

# Select relevant numerical features for the RNN input
selected_features = [
    'packet_length', 'inter_packet_time', 'flow_duration', 
    'total_packets_in_flow', 'total_bytes_in_flow', 'mean_packet_size', 
    'mean_inter_arrival_time', 'packets_per_second', 'bytes_per_second'
]
X = df[selected_features].values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['detailed_label'])
num_classes = len(label_encoder.classes_)
#y = df['detailed_label'].values

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # assuming it's multi-class
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
input_size = X_train.shape[1]  # Number of features
hidden_size = 64  # Reduced hidden size for memory
num_classes = len(set(y))  # Number of unique classes in 'detailed_label'
model = RNNModel(input_size, hidden_size, num_classes).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set hyperparameters
batch_size = 32  # Try reducing this if the error persists
epochs = 20
accumulation_steps = 4  # Gradient accumulation for larger batches if needed

# DataLoader for batching
X_train_tensor = X_train_tensor.unsqueeze(1)  # shape becomes [batch_size, 1, num_features]
X_test_tensor = X_test_tensor.unsqueeze(1)
train_loader = DataLoader(list(zip(X_train_tensor, y_train_tensor)), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(list(zip(X_test_tensor, y_test_tensor)), batch_size=batch_size, shuffle=False)

# Training Loop with memory optimizations
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for i, (inputs, labels) in enumerate(train_loader):
        # Move data to device (GPU if available)
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss = loss / accumulation_steps  # Normalize loss for gradient accumulation

        # Backward pass
        loss.backward()

        if (i + 1) % accumulation_steps == 0:  # Update weights every `accumulation_steps` mini-batches
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item()

    # Print training loss for the epoch
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}")

    # Early stopping (optional)
    if running_loss < 0.1:  # You can define a stopping threshold
        print("Early stopping triggered")
        break

# Save model
torch.save(model.state_dict(), "D:/jenny/Documents/FAUS_Study/Thesis/RNN_Transformer/Models/Standard_RNN_multi_classifier1.pth")
joblib.dump(label_encoder, 'label_encoder.pkl')

# Evaluation
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Compute Confusion Matrix (for multi-class classification)
from sklearn.metrics import confusion_matrix, classification_report
print("Confusion Matrix:\n", confusion_matrix(true_labels, predictions))
cm = confusion_matrix(true_labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix - Multi-Class RNN")
plt.tight_layout()
plt.show()
print("Classification Report:\n", classification_report(true_labels, predictions,target_names=label_encoder.classes_))

