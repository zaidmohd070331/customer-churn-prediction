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
| Logistic Regression (Tuned) | XX% | XX% | XX% | XX% | XX% |
| Random Forest (Tuned)       | XX% | XX% | XX% | XX% | XX% |

🏆 Best Model: [YOUR BEST MODEL NAME HERE]

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

## 🖼️ Key Visualizations
![Feature Importance](plots/feature_importance_final.png)
![Final Model Summary](plots/final_model_summary.png)
![ROC Curves](plots/roc_curves_tuned.png)

## 🚀 How to Run
1. Clone this repository
2. Install dependencies: pip install -r requirements.txt
3. Open churn_prediction.ipynb in Jupyter
4. Run all cells top to bottom

## 📦 Requirements
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.3.0
matplotlib==3.7.0
seaborn==0.12.0
joblib==1.3.0
