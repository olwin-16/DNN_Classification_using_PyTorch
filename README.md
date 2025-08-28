# Deep Neural Network for Breast Cancer Classification

<br>

<img width="1009" height="721" alt="image" src="https://github.com/user-attachments/assets/c5662a44-04f3-4341-baf6-eb186a98f399" />

<br>

## Project Overview

This project builds and trains a deep neural network to classify whether breast tumors are malignant or benign using the Breast Cancer Wisconsin (Diagnostic) Data Set. The project provides hands-on experience with PyTorch and neural network architectures focused on medical diagnosis classification tasks.

## Project Structure

**Main Script:** Single Python file containing data loading, preprocessing, model definition, training, evaluation, and visualization.

**Dataset Handling:** Uses the ucimlrepo package to download and load the dataset.

**Data Preprocessing:** Balances classes, splits data into training and testing sets, and standardizes features.

**Model Architecture:** Defines a shallow neural network with one hidden layer using PyTorch's nn.Module.

**Training & Evaluation:** Implements training loops with CrossEntropyLoss and Adam optimizer, tracks training and test loss, and plots the learning curves.

## Dataset Details

**Source:** [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

**Samples:** 569 entries with 30 numeric features

**Classes:** Benign (B) and Malignant (M)

**Balanced Subset:** 400 samples (200 B and 200 M) used for training/testing

**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

## Data Preprocessing

Data loaded via **fetch_ucirepo(id=17)**

- Created balanced dataset with equal benign and malignant samples

- Converted labels to binary (0 for benign, 1 for malignant)

- Split data into 80% training (320 samples) and 20% testing (80 samples) sets

- Features standardized to zero mean and unit variance using StandardScaler

- Converted data to PyTorch tensors and wrapped in DataLoader for batch processing

## Model Architecture

```bash
import torch.nn as nn

class ClassificationNet(nn.Module):
    def __init__(self, input_units=30, hidden_units=64, output_units=2):
        super(ClassificationNet, self).__init__()
        self.fc1 = nn.Linear(input_units, hidden_units)
        self.fc2 = nn.Linear(hidden_units, output_units)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = ClassificationNet(input_units=30, hidden_units=64, output_units=2)
print(model)
```
- 30 input neurons (features)

- Hidden layer with 64 neurons and ReLU activation

- Output layer with 2 neurons (logits for classes)

##  Training & Optimization

- **Loss function:** CrossEntropyLoss

- **Optimizer:** Adam with learning rate 0.001

- **Number of epochs:** 10

- **Batch size:** 2

During training, loss is logged every epoch, showing steady decrease for both training and test sets.

## Performance Summary

| Epoch | Train Loss | Test Loss |
|-------|------------|-----------|
| 1     | 0.2711     | 0.1905    |
| 10    | 0.0470     | 0.0893    |

- Training and test losses decrease steadily, indicating good model learning and generalization.

- Final test loss is low, and close to training loss, suggesting low overfitting.

## Training and Test Loss Curve

A plot generated using matplotlib shows the training and test loss trend, demonstrating convergence.

<img width="855" height="547" alt="image" src="https://github.com/user-attachments/assets/1a9f9b7b-38f0-4c8d-8c69-84310ccbac92" />

## Setup & Installation

Clone the repository:

```bash
git clone https://github.com/your-username/breast-cancer-classification.git
cd breast-cancer-classification
```

## Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the training script:

```bash
python breast_cancer_classification.py
```

## requirements.txt
```bash
pandas==2.2.2
numpy==1.26.4
matplotlib==3.8.0
scikit-learn==1.5.0
torch==2.3.1
ucimlrepo==0.0.7
```

## License

- The dataset is licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

- The project code is licensed under the MIT License.  
  For details on the MIT License, see [https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT)

## Contact

For questions or contributions, please open an issue or contact via [Email](mailto:olwinchristian@gmail.com)
