import pandas as pd
import os

def create_test_data(normal_file, backdoor_file, total_samples, ratio_normal, ratio_backdoor, output_dir="."):
    """
    Create a test data file by concatenating normal and backdoor data with specified ratios.
    
    Args:
        normal_file (str): Path to the normal data CSV file.
        backdoor_file (str): Path to the backdoor data CSV file.
        total_samples (int): Total number of samples in the test data.
        ratio_normal (float): Ratio of normal data in the test data (e.g., 0.5 for 50%).
        ratio_backdoor (float): Ratio of backdoor data in the test data (e.g., 0.5 for 50%).
        output_dir (str): Directory where the test data file will be saved.
        
    Returns:
        str: Path to the saved test data file.
    """
    # Load the data
    normal_data = pd.read_csv(normal_file)
    backdoor_data = pd.read_csv(backdoor_file)
    print(f"total sample = {total_samples}")
    
    # Calculate the number of samples for each dataset
    num_normal = int(total_samples * ratio_normal)
    num_backdoor = int(total_samples * ratio_backdoor)
    print(num_backdoor)
    print(num_normal)
    
    # Check if there are enough samples in the input files
    if num_normal > len(normal_data):
        raise ValueError(f"Not enough normal data. Requested: {num_normal}, Available: {len(normal_data)}")
    if num_backdoor > len(backdoor_data):
        raise ValueError(f"Not enough backdoor data. Requested: {num_backdoor}, Available: {len(backdoor_data)}")
    
    # Sample the data
    sampled_normal = normal_data.sample(n=num_normal, random_state=42)
    sampled_backdoor = backdoor_data.sample(n=num_backdoor, random_state=42)
    
    # Concatenate the data
    test_data = pd.concat([sampled_normal, sampled_backdoor], ignore_index=True)
    
    # Shuffle the data
    test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Construct the output filename
    output_filename = f"test_data_n{int(ratio_normal * 100)}_backdoor{int(ratio_backdoor * 100)}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save the test data
    test_data.to_csv(output_path, index=False)
    print(f"Test data saved to {output_path}")
    
    return output_path


# Example usage
if __name__ == "__main__":
    # Define input files and parameters
    normal_file = "D:/jenny/Documents/FAUS_Study/Thesis/My_dataset/Normal_data/UNSW-NB15_normal.csv"  # Replace with the path to your normal.csv file
    backdoor_file = "D:/jenny/Documents/FAUS_Study/Thesis/My_dataset/Attacked_data/UNSW-NB15_backdoor_only.csv"  # Replace with the path to your backdoor.csv file
    total_samples = 1000  # Total number of test samples
    ratio_normal = 0.5  # Ratio of normal data
    ratio_backdoor = 0.5  # Ratio of backdoor data
    output_dir = "D:/jenny/Documents/FAUS_Study/Thesis/My_dataset/Attacked_data/Train_dataset/"  # Directory to save the output file
    
    # Create the test data
    create_test_data(normal_file, backdoor_file, total_samples, ratio_normal, ratio_backdoor, output_dir)