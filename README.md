# Multiclass Classification (MNIST) & Polynomial Regression (California Housing)

A comparative machine learning study implementing and critically evaluating multiple supervised learning models on two benchmark problems: multiclass image classification and continuous-value regression.

**Unit:** ITEC203 — Introduction to Data Science and Machine Learning, Australian Catholic University
**Author:** Nishant Shrestha

## Overview

This project applies the core supervised-learning workflow to:

1. **Multiclass classification** on the MNIST handwritten-digit dataset (10 classes, 70,000 images), comparing three algorithms:
   - k-Nearest Neighbours (kNN)
   - Support Vector Machine (SVM) — linear, RBF, and polynomial kernels
   - Feed-forward Neural Network (TensorFlow/Keras)

2. **Polynomial regression** on the California Housing dataset, analysing the bias–variance trade-off across polynomial degrees 1–4.

## Key Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| kNN (k=10) | 0.9716 | 0.9722 | 0.9712 | 0.9716 |
| Neural Network | **0.9784** | 0.9782 | 0.9782 | 0.9782 |
| SVM (linear) | 0.9145 | 0.9151 | 0.9136 | 0.9137 |
| SVM (RBF) | 0.9620 | 0.9620 | 0.9619 | 0.9619 |
| SVM (poly) | 0.9595 | 0.9597 | 0.9591 | 0.9593 |

- The **neural network** achieved the strongest classification performance with fast inference, making it the most production-viable model.
- The **RBF-kernel SVM** gave the best accuracy-to-compute-cost ratio under a constrained training budget.
- For **polynomial regression**, degree 1 (linear) generalised best (R² = 0.61); higher degrees overfit catastrophically (R² collapsed to as low as −6×10⁸), demonstrating the danger of unregularised model complexity.

## Methodology Highlights

- Stratified train/test splitting to preserve class balance
- Data leakage avoided via pipelines (scaling and feature expansion fit only on training data)
- Macro-averaged precision/recall/F1 used alongside accuracy to expose per-class weaknesses
- Accuracy vs. training-time trade-off analysis across SVM kernels
- Bias–variance analysis across polynomial regression degrees
- Data ethics discussion: dataset representativeness, fairness of misclassification impact, and privacy considerations

## Repository Contents

- `mnist_classification_california_housing_regression.ipynb` — full Jupyter notebook with code, outputs, and visualisations
- `Nishant_Shrestha_ML_Classification_Regression_Project.pdf` — formal written report (executive summary, methodology, results, critical discussion, references)

## Tools & Libraries

Python, scikit-learn, TensorFlow/Keras, pandas, NumPy, Matplotlib, Seaborn

## References

- Pedregosa, F. et al. 2011, 'Scikit-learn: Machine learning in Python', *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830.
- Géron, A. 2022, *Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow*, 3rd edn, O'Reilly Media.
- LeCun, Y., Cortes, C. & Burges, C. 1998, *The MNIST Database of Handwritten Digits*.
