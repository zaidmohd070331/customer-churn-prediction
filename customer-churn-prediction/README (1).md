# Customer Churn Prediction — Machine Learning

## 📌 Project Overview
Built an end-to-end ML pipeline to predict customer churn
for a telecom company using Logistic Regression and Random
Forest classifiers with hyperparameter tuning.

## 🛠️ Tools & Libraries
- Python, Pandas, NumPy
- Scikit-learn (Pipeline, GridSearchCV, RandomizedSearchCV)
- Matplotlib, Seaborn

## 📊 Final Model Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (Tuned) | 73.81% | 50.43% | 78.34% | 61.36% | 84.17% |
| Random Forest (Tuned)       | 75.59% | 52.87% | 73.80% | 61.61% | 84.09% |

🏆 Best Model: Random Forest (Tuned)

## 🔍 Key Insights
- Month-to-month contract customers churn 3x more than yearly
- Higher monthly charges strongly correlate with churn
- Customers with more services subscribed are less likely to churn
- Short tenure (0-12 months) has the highest churn rate

## ⚙️ ML Pipeline Steps
1. Exploratory Data Analysis (EDA)
2. Data Cleaning & sklearn Pipeline
3. Feature Engineering (5 new features)
4. Model Training — Logistic Regression + Random Forest
5. Hyperparameter Tuning — GridSearchCV + RandomizedSearchCV
6. Evaluation — Confusion Matrix, ROC Curve, PR Curve
7. Feature Importance Analysis

## 🖼️ Visualizations

### 1. Churn Distribution
![Churn Distribution](churn_distribution.png)

### 2. Numerical Feature Distributions
![Numerical Distributions](numerical_distributions.png)

### 3. Churn Rate by Categorical Features
![Churn by Categories](churn_by_categories.png)

### 4. Correlation Heatmap
![Correlation Heatmap](correlation_heatmap.png)

### 5. Class Imbalance in Training Set
![Class Imbalance](class_imbalance.png)

### 6. Engineered Features vs Churn
![Engineered Features](engineered_features.png)

### 7. Feature Importance — Both Models
![Feature Importance](feature_importance_final.png)

### 8. Confusion Matrices — Tuned Models
![Confusion Matrices](confusion_matrices_tuned.png)

### 9. ROC Curves — Tuned Models
![ROC Curves](roc_curves_tuned.png)

### 10. Precision-Recall Curve
![Precision Recall Curve](precision_recall_curve.png)

### 11. Hyperparameter Tuning Impact
![Tuning Comparison](tuning_comparison.png)

### 12. Final Model Performance Summary
![Final Model Summary](final_model_summary.png)

## 🚀 How to Run
1. Clone this repository
2. Install dependencies: pip install -r requirements.txt
3. Open churn_prediction.ipynb in Jupyter
4. Run all cells top to bottom

## 📦 Requirements
```
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.3.0
matplotlib==3.7.0
seaborn==0.12.0
joblib==1.3.0
```
