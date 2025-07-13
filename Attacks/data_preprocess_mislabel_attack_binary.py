import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv('D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/merged_dataset.csv')

# Check the original distribution of labels
print("Original Label Distribution:")
print(df['label'].value_counts())

# Select the malicious data (assuming 'label' column has 'malicious' and 'benign' labels)
benign_data = df[df['label'] == 'benign']

# Calculate how many malicious entries to change to benign (20% of malicious data)
num_benign_to_change = int(len(benign_data) * 0.20)

# Randomly select 20% of malicious data to change labels
indices_to_change = benign_data.sample(num_benign_to_change, random_state=42).index

# Change the labels of selected malicious data to benign
df.loc[indices_to_change, 'label'] = 'malicious'

# Check the modified distribution of labels
print("\nModified Label Distribution:")
print(df['label'].value_counts())

# Save the modified dataset (if desired)
df.to_csv('D:/jenny/Documents/FAUS_Study/Thesis/My_IoT_23/mislabelled_dataset_20.csv', index=False)

