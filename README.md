# darwin
Early Detection of Alzheimer’s Using Handwriting Data 

This work presents a machine learning approach to detect Alzheimer’s Disease using the DARWIN dataset, which includes handwriting data from 174 individuals (89 with Alzheimer’s, 85 healthy).

Goal: Build a non-invasive, low-cost classification model based on handwriting features.

Steps: Feature extraction → SMOTE balancing → RFE & drop-column selection → Ensemble model (Gradient Boosting, Random Forest, Logistic Regression) → Threshold optimization

Evaluation: Confusion matrix, ROC & precision-recall curves

✅ Results
Accuracy: 94%
F1 Score: 0.95
Cross-validation Accuracy: ~76%

Handwriting-based models like this could support early Alzheimer’s diagnosis in accessible ways.

