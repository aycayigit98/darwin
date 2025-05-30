 Alzheimer’s Detection Using Handwriting Data (DARWIN Dataset)
This project presents a machine learning approach for the early detection of Alzheimer’s disease using handwriting data from the [DARWIN dataset (Diagnosis AlzheimeR WIth haNdwriting)].

The dataset contains handwriting signals from 174 individuals (89 Alzheimer’s patients, 85 healthy), recorded across 25 writing tasks. After feature extraction and preprocessing, a stacked ensemble classifier was trained using:

  -Gradient Boosting

  -Random Forest

  -Logistic Regression

📌 Key Steps
Feature engineering from time-series signals (mean, std, variance)

Feature selection using RFE and Drop Column

Class balancing with SMOTE

Ensemble classification & threshold optimization

Evaluation with confusion matrix, ROC, and Precision-Recall curves

📊 Results
Accuracy: 94%

F1 Score: 0.95

Recall (Patients): 1.00

Cross-validation Avg Accuracy: 76%

📁 Files
darwin.py: Main code with preprocessing, model training, and evaluation.

README.md: Project summary.

