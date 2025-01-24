import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# Load segregated device data paths
device_paths = [f"device_data/device_{i}.csv" for i in range(1, 42)]  # Example paths for 41 devices

# Helper function to load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    # Define required features
    features = [
        "dur", "sbytes", "dbytes", "Sload", "Dload", "Spkts", "Dpkts", 
        "smeansz", "dmeansz", "sloss", "dloss", "Sintpkt", "Dintpkt", 
        "swin", "dwin", "tcprtt", "synack", "ackdat", 
        "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm"
    ]
    
    # Ensure the dataset contains all required features
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        print(f"Warning: Missing features {missing_features} in {file_path}. Skipping this device.")
        return None, None, None, None
    
    X = data[features]
    y = data.get("Label", None)  # Ensure "Label" column exists
    if y is None:
        print(f"Warning: 'Label' column is missing in {file_path}. Skipping this device.")
        return None, None, None, None
    
    # Handle small datasets by enforcing a minimum number of samples
    if len(X) < 10:  # Arbitrary threshold for minimum samples
        print(f"Warning: Insufficient data in {file_path}. Skipping this device.")
        return None, None, None, None
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into training and testing datasets
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e:
        print(f"Error during train-test split for {file_path}: {e}")
        return None, None, None, None
    
    return X_train, X_test, y_train, y_test

# Define the neural network model
def create_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer=SGD(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Federated Learning Simulation
global_weights = None
n_rounds = 1  # One round for this step
device_models = []
aggregated_weights = None

for round_num in range(n_rounds):
    print(f"--- Round {round_num + 1} ---")
    client_updates = []
    
    for device_idx, device_path in enumerate(device_paths):
        print(f"Training on device {device_idx + 1}...")
        
        # Load and preprocess device data
        X_train, X_test, y_train, y_test = load_and_preprocess_data(device_path)
        if X_train is None:  # Skip if data loading failed
            continue
        
        # Create and train the model
        input_dim = X_train.shape[1]
        model = create_model(input_dim)
        
        if global_weights is not None:
            model.set_weights(global_weights)  # Load global weights
        
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
        client_updates.append(model.get_weights())
        device_models.append(model)
    
    # Skip aggregation if no clients contributed
    if not client_updates:
        print("No clients contributed in this round. Exiting.")
        break
    
    # Federated Averaging: Aggregate weights
    print("Aggregating updates...")
    aggregated_weights = [np.mean([client_weights[layer] for client_weights in client_updates], axis=0)
                          for layer in range(len(client_updates[0]))]
    global_weights = aggregated_weights  # Set new global weights
    
    # Update device models with the new global weights
    for model in device_models:
        model.set_weights(global_weights)

print("Federated Learning Round Complete!")
