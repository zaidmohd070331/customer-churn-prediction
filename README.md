# Customer Churn Prediction — Machine Learning

## 📌 Project Overview
Built an end-to-end ML pipeline to predict customer churn for a telecom company
using Logistic Regression and Random Forest classifiers with hyperparameter tuning.
The dataset contains 7,043 customers with a churn rate of 26.5%.

## 🛠️ Tools & Libraries
- Python, Pandas, NumPy
- Scikit-learn (Pipeline, GridSearchCV, RandomizedSearchCV)
- Matplotlib, Seaborn

## 📊 Final Model Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (Tuned) | 73.81% | 50.43% | 78.34% | 61.36% | 84.17% |
| Random Forest (Tuned)       | 75.59% | 52.87% | 73.80% | 61.61% | 84.09% |

🏆 **Best Model: Random Forest (Tuned)**
- Tuned using RandomizedSearchCV with 30 iterations across 5-fold StratifiedKFold
- Best Parameters: n_estimators=200, max_depth=10, min_samples_leaf=4, max_features=log2
- Best CV F1 Score: 63.33% | Std Dev: 1.48%

## 🔍 Key Insights
- Month-to-month contract customers churn 3x more than yearly contract customers
- Higher monthly charges strongly correlate with churn behavior
- Customers with more subscribed services are less likely to churn
- Short tenure (0–12 months) has the highest churn rate — first year is most critical
- 26.5% overall churn rate — dataset is imbalanced, handled via class_weight='balanced'

## ⚙️ ML Pipeline Steps
1. Exploratory Data Analysis (EDA) — churn distribution, feature correlations, category-wise churn rates
2. Data Cleaning & sklearn Pipeline — null handling, encoding, StandardScaler via ColumnTransformer
3. Feature Engineering — 5 new features: Avg_Monthly_Spend, Tenure_Group, Total_Services, Is_LongTerm_Contract, Has_Premium_Services
4. Model Training — Logistic Regression (0.21s) + Random Forest (0.35s) with class_weight='balanced'
5. Hyperparameter Tuning — GridSearchCV for LR | RandomizedSearchCV (30 iter) for RF
6. Evaluation — Confusion Matrix, ROC Curve, Precision-Recall Curve
7. Feature Importance Analysis — RF importances + LR absolute coefficients

## 🖼️ Key Visualizations
![Feature Importance](plots/feature_importance_final.png)
![Final Model Summary](plots/final_model_summary.png)
![ROC Curves](plots/roc_curves_tuned.png)

## 🚀 How to Run
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Open `churn_prediction.ipynb` in Jupyter
4. Run all cells top to bottom

## 📦 Requirements
