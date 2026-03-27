# Automated Loan Approval System using Artificial Neural Networks

Link dataset: [Kaggle](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)

## Project Overview
This mini project contains an end-to-end Deep Learning pipeline designed to predict loan approval status. Built using a Keras Sequential Artificial Neural Network (ANN), this project focuses on effectively processing raw financial data, handling class imbalance, and delivering robust binary classification metrics.

## Data Pipeline & Preprocessing
To ensure the neural network learns optimal patterns, the following preprocessing steps were implemented:
1. **Feature Engineering:** Renamed columns for better readability (e.g., `cb_person_cred_hist_length` to `credit_age`).
2. **Categorical Encoding:** Applied `LabelEncoder` for categorical variables (education, home ownership, loan intent) and mapped binary features.
3. **Scaling:** Utilized `StandardScaler` to normalize numerical features.
4. **Handling Class Imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) strictly on the training set to prevent model bias towards the majority class (Approved loans).

## Model Architecture
The model is built using a sequential fully connected neural network with dropout regularization to prevent overfitting:
* **Input Layer:** Scaled features (StandardScaler).
* **Hidden Layer 1:** 128 neurons (`tanh` activation) + 0.3 Dropout.
* **Hidden Layer 2:** 100 neurons (`tanh` activation) + 0.2 Dropout.
* **Hidden Layer 3:** 64 neurons (`tanh` activation) + 0.1 Dropout.
* **Output Layer:** 1 neuron (`sigmoid` activation) for binary classification (0 = Rejected, 1 = Approved).
*Training techniques applied include **Early Stopping** (patience=10) and **Class Weights** to further penalize misclassifications of the minority class.*

## Results & Evaluation

The model performs exceptionally well, achieving a balanced capability in predicting both approved and rejected loans.

### 1. Classification Metrics
* **Accuracy:** 91.18%
* **ROC-AUC Score:** 87.07%

```text
              precision    recall  f1-score   support

           0       0.94      0.94      0.94      2100
           1       0.80      0.80      0.80       600

    accuracy                           0.91      2700
   macro avg       0.87      0.87      0.87      2700
weighted avg       0.91      0.91      0.91      2700
