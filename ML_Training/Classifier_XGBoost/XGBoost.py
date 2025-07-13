import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (training)
features = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/CTU-IoT-Malware-Capture-1-1/2018-05-09-192.168.100.103.horizontalPort_packet_flow_features_merged_labeled.csv"
df = pd.read_csv(features)  # Replace with your dataset file path

# Drop unnecessary columns
drop_cols = ["ts", "formatted_time", "src_ip", "dst_ip", "detailed_label"]
df.drop(columns=drop_cols, inplace=True)

# Convert categorical features to numeric
categorical_cols = ["proto", "tcp_flags"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save for future decoding

# Convert labels to binary (Benign = 0, Malicious = 1)
df["label"] = df["label"].apply(lambda x: 0 if x == "Benign" else 1)

# Separate features and labels
X = df.drop(columns=["label"])
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Ensure features are float before applying SMOTE
X_train = X_train.astype(float)
X_test = X_test.astype(float)

# Handle class imbalance using SMOTE
smote = SMOTE(sampling_strategy=0.75, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train XGBoost classifier
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=2.3)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Benign", "Malicious"], yticklabels=["Benign", "Malicious"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Save model and scaler
joblib.dump(model, "iot23_xgb_binary_model.pkl")
joblib.dump(scaler, "iot23_binary_scaler.pkl")

def evaluate_new_data(file_path):
    # Load new dataset for evaluation
    new_df = pd.read_csv(file_path)

    # Preprocessing for new data (similar to training data preprocessing)
    new_df.drop(columns=drop_cols, inplace=True)
    
    for col in categorical_cols:
        new_df[col] = label_encoders[col].transform(new_df[col])
    
    new_df.fillna(0, inplace=True)

    # Separate features from labels (if 'label' column exists)
    X_new = new_df.drop(columns=["label"], errors='ignore')  # Drop label column if it exists

    # Ensure that the new data has the same feature columns as the training data
    missing_cols = set(X.columns) - set(X_new.columns)
    for col in missing_cols:
        X_new[col] = 0  # Add missing columns with zero values

    X_new = X_new[X.columns]  # Ensure the same order of columns as in training data

    # Normalize using the saved scaler
    new_data = scaler.transform(X_new)

    # Predict using the trained model
    predictions = model.predict(new_data)

    # Add predictions to the new dataframe for evaluation
    new_df["predicted_label"] = ["Benign" if pred == 0 else "Malicious" for pred in predictions]
    
    # If the new dataset has labels, evaluate the performance
    if "label" in new_df.columns:
        y_new = new_df["label"]
        y_true_numeric = np.where(y_new == "Benign", 0, 1)
        print("\nEvaluation on New Dataset:")
        accuracy = accuracy_score(y_true_numeric, predictions)
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_true_numeric, predictions))

        # Confusion Matrix for new data
        conf_matrix_new = confusion_matrix(y_true_numeric, predictions)
        print("Confusion Matrix (New Dataset):")
        print(conf_matrix_new)

        # Plot Confusion Matrix for new data
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix_new, annot=True, fmt='d', cmap='Blues', xticklabels=["Benign", "Malicious"], yticklabels=["Benign", "Malicious"])
        plt.title("Confusion Matrix (New Dataset)")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

    return new_df

# Example usage for a new evaluation CSV
new_data_file = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/CTU-IoT-Malware-Capture-8-1/2018-07-31-15-15-09-192.168.100.113_packet_flow_features_labeled.csv"  # Replace with your new dataset file path
new_evaluated_data = evaluate_new_data(new_data_file)
print(new_evaluated_data.head())  # Show first few rows of predictions

'''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load dataset
features = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/CTU-IoT-Malware-Capture-1-1/2018-05-09-192.168.100.103.horizontalPort_packet_flow_features_merged_labeled.csv"
df = pd.read_csv(features)  # Change filename if needed

# Drop unnecessary columns
drop_cols = ["ts", "formatted_time", "src_ip", "dst_ip", "detailed_label"]
df.drop(columns=drop_cols, inplace=True)

# Convert categorical features to numeric
categorical_cols = ["proto", "tcp_flags"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save for future decoding

# Convert labels to binary (Benign = 0, Malicious = 1)
df["label"] = df["label"].apply(lambda x: 0 if x == "Benign" else 1)

# Separate features and labels
X = df.drop(columns=["label"])
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Ensure features are float before applying SMOTE
X_train = X_train.astype(float)
X_test = X_test.astype(float)

# Handle class imbalance using SMOTE
smote = SMOTE(sampling_strategy=0.75, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train XGBoost classifier
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=2.3)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model and scaler
import joblib
joblib.dump(model, "iot23_xgb_binary_model.pkl")
joblib.dump(scaler, "iot23_scaler.pkl")

# Function to predict new samples
def predict_new(sample_df):
    sample_df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Encode categorical features
    for col in categorical_cols:
        sample_df[col] = label_encoders[col].transform(sample_df[col])

    sample_df.fillna(0, inplace=True)
    sample_df = scaler.transform(sample_df)
    prediction = model.predict(sample_df)
    return "Benign" if prediction[0] == 0 else "Malicious"

# Example usage:
# new_sample = pd.DataFrame([{ "proto": "TCP", "packet_length": 400, "flow_duration": 2.5, ... }])
# print(predict_new(new_sample))
'''