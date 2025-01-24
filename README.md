# Federated Learning and Poisoning Attacks on IoT Devices

Welcome to the repository for my thesis project, *"Analyzing and Defending Against Poisoning Attacks on IoT Devices Using Federated Learning."* This project explores the intersection of Federated Learning (FL) and adversarial machine learning, focusing on how poisoning attacks impact IoT systems and proposing defense mechanisms to mitigate such attacks.

---

## **Overview**
IoT devices are widely deployed across various industries, from smart homes to healthcare and beyond. These devices continuously collect and share data, often relying on machine learning to make intelligent decisions. However, their interconnected nature makes them vulnerable to attacks.

Federated Learning (FL) is a decentralized approach to training machine learning models where data remains on the edge devices, ensuring privacy and reducing communication overhead. While this approach offers several benefits, it is also susceptible to poisoning attacks where adversaries introduce malicious data to compromise the model's performance.

This project demonstrates:
1. How poisoning attacks impact Federated Learning models in IoT environments.
2. Defense mechanisms to mitigate these attacks and ensure model robustness.

---

## **Dataset**
The project uses the **UNSW-NB15 dataset**, a benchmark dataset for evaluating intrusion detection systems. This dataset contains records of normal network activities and 9 types of cyberattacks.

**Dataset Details:**
- **Normal Traffic**: Legitimate traffic without attacks.
- **Malicious Traffic**: Includes attacks such as backdoor, SQL injection, and more.
- **Segregation by Source IP**: IoT devices are simulated by segregating data based on unique `srcip` (source IP addresses).

---

## **Project Structure**

```
📁 root directory
├── 📂 data
│   ├── UNSW-NB15_1.csv
│   ├── UNSW-NB15_2.csv
│   ├── UNSW-NB15_3.csv
│   ├── UNSW-NB15_4.csv
│   └── device-specific datasets
├── 📂 scripts
│   ├── 📂 Data_preprocessing
|   |      ├── Header_concation            # Script to add headers to the raw data file using the feature file.
|   |      ├── Normal_Attacked_classifier  # Script to seperate normal and compromised data.
|   |      └── data_segmentation.py        # Script to segregate data by srcip.
│   ├── 📂 Federated_Learning
|   |      └── federated_training.py       # FL implementation.
│   ├── 📂 Attack_Simulator
|   |      └── poisoning_attack_sim.py     # Simulates poisoning attacks.
│   ├── 📂 Evaluation
|   |      └── evaluation_metrics.py       # Computes evaluation metrics.
│   └── 📂 Defense_Algorithms
|   |      └── defenses.py                 # Defense mechanisms.
├── 📂 models
│   ├── global_model.pkl
│   └── device_model_<device_id>.pkl
├── 📂 results
│   ├── confusion_matrix_device.png
│   ├── evaluation_report_device_before_attack.txt
|   └──evalution_report_device_after_attack.txt
├── README.md                     # Current file
└── requirements.txt              # Dependencies
```

---

## **Approach**

### 1. **Data Preprocessing**
   - Combined UNSW-NB15_1 to UNSW-NB15_4 datasets.
   - Segregated traffic by `srcip` to simulate IoT devices in a smart home environment.
   - Created a balanced distribution of normal and malicious traffic for each device.

### 2. **Federated Learning Implementation**
   - Trained local models on device-specific data.
   - Aggregated local model updates into a global model using Federated Averaging (FedAvg).

### 3. **Simulating Poisoning Attacks**
   - Injected compromised nodes where malicious data (20%) was mixed into the training data.
   - Evaluated the impact of these poisoned updates on the global model.

### 4. **Defense Mechanisms**
   - **Data Filtering**: Excluded anomalous updates based on statistical metrics.
   - **Robust Aggregation**: Used methods like Trimmed Mean and Krum to minimize the influence of poisoned updates.
   - **Thresholding**: Implemented stricter thresholds for model update acceptance.

### 5. **Evaluation**
   - Metrics: Precision, Recall, F1-score, Accuracy.
   - Tools: Confusion matrices and classification reports for each device.

---

<!--## **Key Results**
1. **Without Defense**:
   - The global model was heavily influenced by poisoned updates, leading to reduced accuracy across all devices.

2. **With Defense**:
   - Data filtering and robust aggregation significantly mitigated the impact of poisoned updates.
   - Improved F1-scores and recall for detecting malicious traffic.

| Metric         | Without Defense | With Defense |
|----------------|-----------------|--------------|
| Precision      | 0.45           | 0.78         |
| Recall         | 0.30           | 0.72         |
| F1-Score       | 0.36           | 0.75         |

--- -->

## **Setup and Usage**

### **Requirements**
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### **Running the Scripts**
1. **Data Segmentation**
   ```bash
   python scripts/data_segmentation.py
   ```
   This script creates device-specific datasets from the combined UNSW-NB15 dataset.

2. **Training Federated Learning Models**
   ```bash
   python scripts/federated_training.py
   ```

3. **Simulating Poisoning Attacks**
   ```bash
   python scripts/poisoning_attack_sim.py
   ```

4. **Evaluating Model Performance**
   ```bash
   python scripts/evaluation_metrics.py
   ```

---

<!--## **Contributing**

Contributions are welcome! Feel free to submit issues or pull requests to improve this project.

---

## **Acknowledgments**

Special thanks to my professor and mentors for their guidance throughout this project. The UNSW-NB15 dataset creators are also acknowledged for providing a valuable resource for this research.

---

## **License**

This project is licensed under the MIT License. See the LICENSE file for more details.

---

## **Contact**

If you have any questions or suggestions regarding this project, feel free to reach out:
- Email: [jennynadar9@gmail.com]
- LinkedIn: [https://www.linkedin.com/in/jenny-nadar-794533161/]
-->
