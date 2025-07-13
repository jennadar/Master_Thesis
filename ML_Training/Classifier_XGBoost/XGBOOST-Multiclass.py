import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
def load_and_preprocess_data(features_file):
    df = pd.read_csv(features_file)

    # Encode 'label' (Binary: Benign = 0, Malicious = 1)
    df['label'] = df['label'].map({'Benign': 0, 'Malicious': 1})

    # Encode 'detailed_label' (Multi-Class)
    label_encoder = LabelEncoder()
    df['detailed_label'] = label_encoder.fit_transform(df['detailed_label'])

    # Drop unnecessary columns (IP addresses, timestamps, etc.)
    drop_cols = ['ts', 'formatted_time', 'src_ip', 'dst_ip']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # One-Hot Encoding for 'proto' and 'tcp_flags'
    df = pd.get_dummies(df, columns=['proto', 'tcp_flags'], drop_first=True)

    # Define Features (X) and Target (y)
    X = df.drop(columns=['detailed_label'])  # Features
    y = df['detailed_label']  # Target

    # Normalize numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler, label_encoder


# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, label_encoder):
    y_true_decoded = label_encoder.inverse_transform(y_true)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    # Generate confusion matrix
    cm = confusion_matrix(y_true_decoded, y_pred_decoded)
    # Create a DataFrame for better visualization
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, fmt=".2f", cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix of Multiclass xgboost')
    #plt.savefig("confusion_matrix_multiclass_xgboost.png")
    plt.show()

# Train and evaluate XGBoost model
def train_xgboost(features_file):
    X, y, scaler, label_encoder = load_and_preprocess_data(features_file)

    # Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # XGBoost Classifier
    model = xgb.XGBClassifier(objective="multi:softmax", num_class=len(np.unique(y)), eval_metric="mlogloss")
    model.fit(X_train, y_train)

    # Predictions on Test Data
    y_pred = model.predict(X_test)

    # Model Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\n Classification Report (Test Set):")
    print(classification_report(y_test, y_pred, zero_division=1))

    print("\n Confusion Matrix (Test Set):")
    print(confusion_matrix(y_test, y_pred))
    plot_confusion_matrix(y_test,y_pred,label_encoder)

    # Save model and scaler
    joblib.dump(model, "iot23_xgb_multiclass_model.pkl")
    joblib.dump(scaler, "iot23_xgb_multiclass_scaler.pkl")

    return model, scaler, label_encoder


# Evaluate on Unseen Dataset
def evaluate_on_unseen_data(model, scaler, label_encoder, unseen_file):
    X_unseen, y_unseen, _, _ = load_and_preprocess_data(unseen_file)

    # Predict on Unseen Data
    y_pred_unseen = model.predict(X_unseen)

    # Model Evaluation
    accuracy = accuracy_score(y_unseen, y_pred_unseen)
    print(f"Accuracy: {accuracy:.4f}")
    print("\n Classification Report (Unseen Dataset):")
    print(classification_report(y_unseen, y_pred_unseen, zero_division=1))

    print("\n Confusion Matrix (Unseen Dataset):")
    print(confusion_matrix(y_unseen, y_pred_unseen))
    plot_confusion_matrix(y_unseen,y_pred_unseen,label_encoder)

# Run the Model
if __name__ == "__main__":
    train_file = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/merged_dataset.csv"  # Update this path
    unseen_file = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/merged_dataset1.csv"  # Update this path

    # Train Model
    model, scaler, label_encoder = train_xgboost(train_file)

    # Test on Unseen Data
    evaluate_on_unseen_data(model, scaler, label_encoder, unseen_file)


'''import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
def load_and_preprocess_data(features_file):
    df = pd.read_csv(features_file)

    # Encode 'label' (Binary: Benign = 0, Malicious = 1)
    df['label'] = df['label'].map({'Benign': 0, 'Malicious': 1})

    # Encode 'detailed_label' (Multi-Class)
    label_encoder = LabelEncoder()
    df['detailed_label'] = label_encoder.fit_transform(df['detailed_label'])

    # Drop unnecessary columns (IP addresses, timestamps, categorical)
    drop_cols = ['ts', 'formatted_time', 'src_ip', 'dst_ip', 'proto']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Convert non-numeric columns to numeric if necessary
    df = df.apply(pd.to_numeric, errors='ignore')

    # Define Features (X) and Target (y)
    X = df.drop(columns=['detailed_label'])  # Features
    y = df['detailed_label']  # Target

    # Normalize numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler, label_encoder

# Calculate class weights based on class distribution
def calculate_class_weights(y):
    class_weights = y.value_counts(normalize=True)
    weight_dict = {label: 1.0 / weight for label, weight in class_weights.items()}
    return weight_dict

# Train and evaluate XGBoost model
def train_xgboost(features_file):
    X, y, scaler, label_encoder = load_and_preprocess_data(features_file)

    # Calculate class weights
    class_weights = calculate_class_weights(y)

    # Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Assign sample weights to each class
    sample_weights = y_train.map(class_weights).values

    # XGBoost Classifier
    model = xgb.XGBClassifier(objective="multi:softmax", num_class=len(np.unique(y)), eval_metric="mlogloss")
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Predictions on Test Data
    y_pred = model.predict(X_test)

    # Model Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, zero_division=1))

    print("\nConfusion Matrix (Test Set):")
    print(confusion_matrix(y_test, y_pred))

    # Save model and scaler
    joblib.dump(model, "iot23_xgb_multiclass_model.pkl")
    joblib.dump(scaler, "iot23_xgb_multiclass_scaler.pkl")

    # Plot Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, label_encoder)

    return model, scaler, label_encoder

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, label_encoder):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize confusion matrix

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.show()

# Evaluate on Unseen Dataset
def evaluate_on_unseen_data(model, scaler, label_encoder, unseen_file):
    X_unseen, y_unseen, _, _ = load_and_preprocess_data(unseen_file)

    # Predict on Unseen Data
    y_pred_unseen = model.predict(X_unseen)

    # Model Evaluation
    accuracy = accuracy_score(y_unseen, y_pred_unseen)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report (Unseen Dataset):")
    print(classification_report(y_unseen, y_pred_unseen, zero_division=1))

    print("\nConfusion Matrix (Unseen Dataset):")
    print(confusion_matrix(y_unseen, y_pred_unseen))

    # Plot Confusion Matrix
    plot_confusion_matrix(y_unseen, y_pred_unseen, label_encoder)

# Run the Model
if __name__ == "__main__":
    train_file = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/merged_dataset.csv"  # Update this path
    unseen_file = "D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/merged_dataset1.csv"  # Update this path

    # Train Model
    model, scaler, label_encoder = train_xgboost(train_file)

    # Test on Unseen Data
    evaluate_on_unseen_data(model, scaler, label_encoder, unseen_file)
'''