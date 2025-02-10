# Import necessary libraries
import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sagemaker
from sagemaker import get_execution_role
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, mean_absolute_error, mean_squared_error, r2_score)

# Get SageMaker execution role
role = get_execution_role()

# Define S3 bucket and file paths
s3_bucket = "projectsagemaker"  # Replace with your bucket name
train_s3_path = "s3://projectsagemaker/train.csv"  # Train dataset
test_s3_path = "s3://projectsagemaker/test.csv"  # Test dataset

# Use boto3 to load data from S3 into a Pandas DataFrame
s3 = boto3.client("s3")


def load_s3_csv(s3_path):
    """Load CSV file from S3 into Pandas DataFrame"""
    bucket_name = s3_path.split("/")[2]
    file_key = "/".join(s3_path.split("/")[3:])

    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    return pd.read_csv(obj["Body"])


# Load data from S3
train_data = load_s3_csv(train_s3_path).sample(n=10000, random_state=42)
test_data = load_s3_csv(test_s3_path).sample(n=2000, random_state=42)

# Drop duplicates
train_data.drop_duplicates(keep="first", inplace=True)
test_data.drop_duplicates(keep="first", inplace=True)

# Define classification feature matrix and target
classification_column = "target"
X_class = train_data.drop(columns=["ID_code", classification_column])
y_class = train_data[classification_column]

# Train-test split
X_train_class, X_val_class, y_train_class, y_val_class = train_test_split(X_class, y_class, test_size=0.2,
                                                                          random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_class_scaled = scaler.fit_transform(X_train_class)
X_val_class_scaled = scaler.transform(X_val_class)

# Classification models
classification_models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=50, max_depth=10, random_state=42, use_label_encoder=False,
                             eval_metric='logloss')
}

# Train and evaluate models
classification_metrics = {}

for model_name, model in classification_models.items():
    if model_name == "Logistic Regression":
        model.fit(X_train_class_scaled, y_train_class)
        y_pred_class = model.predict(X_val_class_scaled)
        y_pred_proba_class = model.predict_proba(X_val_class_scaled)[:, 1]
    else:
        model.fit(X_train_class, y_train_class)
        y_pred_class = model.predict(X_val_class)
        y_pred_proba_class = model.predict_proba(X_val_class)[:, 1]

    accuracy = accuracy_score(y_val_class, y_pred_class)
    precision = precision_score(y_val_class, y_pred_class)
    recall = recall_score(y_val_class, y_pred_class)
    f1 = f1_score(y_val_class, y_pred_class)
    roc_auc = roc_auc_score(y_val_class, y_pred_proba_class)

    classification_metrics[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC-ROC": roc_auc
    }
    print(
        f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC-ROC: {roc_auc:.4f}")

# Convert to DataFrame
classification_metrics_df = pd.DataFrame(classification_metrics).T
print("\nClassification Model Performance Comparison:")
print(classification_metrics_df.to_string())

# Select best model
best_class_model_name = max(classification_metrics, key=lambda x: classification_metrics[x]['AUC-ROC'])
final_class_model = classification_models[best_class_model_name]

# Train on full dataset
if best_class_model_name == "Logistic Regression":
    final_class_model.fit(scaler.fit_transform(X_class), y_class)
    y_test_pred_class = final_class_model.predict(scaler.transform(test_data.drop(columns="ID_code")))
else:
    final_class_model.fit(X_class, y_class)
    y_test_pred_class = final_class_model.predict(test_data.drop(columns="ID_code"))

print("Test Predictions (Classification):", y_test_pred_class)

# Revenue Forecasting
train_data['revenue'] = train_data.iloc[:, 2:].sum(axis=1)
test_data['revenue'] = test_data.iloc[:, 2:].sum(axis=1)

X_reg = train_data.drop(columns=["ID_code", "target", "revenue"])
y_reg = train_data["revenue"]

X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

regression_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
}

regression_metrics = {}

for model_name, model in regression_models.items():
    model.fit(X_train_reg, y_train_reg)
    y_pred_reg = model.predict(X_val_reg)

    mae = mean_absolute_error(y_val_reg, y_pred_reg)
    mse = mean_squared_error(y_val_reg, y_pred_reg)
    r2 = r2_score(y_val_reg, y_pred_reg)

    regression_metrics[model_name] = {
        "MAE": mae,
        "MSE": mse,
        "R^2": r2
    }
    print(f"{model_name} - MAE: {mae:.4f}, MSE: {mse:.4f}, R^2: {r2:.4f}")

regression_metrics_df = pd.DataFrame(regression_metrics).T
print("\nRegression Model Performance Comparison:")
print(regression_metrics_df.to_string())