# 🤖 Machine Learning Mini-Projects

This repository demonstrates a collection of supervised learning tasks using different datasets.
The focus is on Logistic Regression (binary classification) and k-Nearest Neighbors (classification & regression) applied to real-world inspired problems.

## 📂 Datasets

Each dataset targets a different ML problem:

📧 Email Spam Detection

👥 Customer Churn Prediction

🌸 Flower Classification (Iris-style)

🏠 Airbnb Price Prediction

## 🔹 Problem Statements & Tasks

1) 📧 Email Spam — Logistic Regression (Binary)

Objective: Detect spam emails using logistic regression.

Dataset columns:
word_free, word_offer, word_click, num_links, num_caps, sender_reputation, is_spam

Tasks:

Fit is_spam ~ features using Logistic Regression with scaling.

Evaluate with multiple metrics.

Metrics reported:
Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix

2) 👥 Customer Churn — Logistic Regression (Binary)

Objective: Predict whether a customer will churn based on usage and subscription features.

Dataset columns:
tenure_months, monthly_charges, support_tickets, is_premium, avg_usage_hours, churn

Tasks:

Fit logistic regression model on customer churn data.

Apply feature scaling before training.

Metrics reported:
Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix

3) 🌸 Flowers — k-NN Classification with Cross-Validation

Objective: Classify flowers (species) based on their measurements.

Dataset columns:
sepal_length, sepal_width, petal_length, petal_width, species

Tasks:

Train k-NN classifier using features.

Use 5-fold CV to select best k ∈ {1,3,…,25}.

Metrics reported:
Best k, CV Score, Test Accuracy, Confusion Matrix

4) 🏠 Airbnb Prices — k-NN Regression with Cross-Validation

Objective: Predict Airbnb prices based on property features.

Dataset columns:
size_m2, distance_center_km, rating, num_reviews, price

Tasks:

Train k-NN regressor with scaled features.

Use 5-fold CV to tune best k ∈ {1,3,…,25}.

Metrics reported:
CV RMSE, Test RMSE, Test R²

## ⚙️ Methods

Load datasets using Pandas

Preprocess data with feature scaling

Train models with scikit-learn

Evaluate with standard metrics

Visualize confusion matrices and residuals where applicable

## 📊 Results

✅ Logistic Regression performed well on binary classification tasks (spam, churn)

✅ k-NN with cross-validation helped select the best k for flowers and Airbnb datasets

✅ Evaluation metrics (Accuracy, F1, RMSE, R²) provided clear insights into performance

## 📦 Requirements

Install dependencies with:

pip install pandas numpy scikit-learn matplotlib seaborn


## 📈 Key Insights

Logistic Regression is simple yet effective for binary problems

k-NN requires careful selection of k (cross-validation is crucial)

Feature scaling has a big impact on model performance

Metrics beyond accuracy (F1, RMSE, R²) give deeper evaluation

## 🚀 Future Work

🔹 Extend to more datasets (healthcare, finance, etc.)

🔹 Try advanced models (Random Forests, SVMs, Gradient Boosting)

🔹 Add hyperparameter tuning (GridSearchCV)

🔹 Visualize results with interactive plots
