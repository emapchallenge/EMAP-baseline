# -*- coding: utf-8 -*-
"""
Created on Sat Aug 6 12:33:12 2022
@author: harisushehu
"""

# Import Libraries
import os
import csv
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import pyswarms as ps
from glob import glob

# Constants
TRAIN_DIR = "./data/train"
VAL_DIR = "./data/val"
RESULTS_FILE = "./results/classificationArousal_DT.csv"
SELECTED_FEATURES_FILE = "./results/classificationArousal_selected_features.csv"
FEATURES_THRESHOLD_FILE = "./results/classificationArousal_features.csv"
MODEL_SAVE_PATH = "./models"
initial_model_path = os.path.join(MODEL_SAVE_PATH, "classificationArousal_DT_initial.pkl")
selected_model_path = os.path.join(MODEL_SAVE_PATH, "classificationArousal_DT_selected_features.pkl")


# Helper Functions
def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = csv.writer(write_obj)
        csv_writer.writerow(list_of_elem)

def save_selected_features(file_name, selected_features):
    """Save selected features' indices to a CSV file."""
    with open(file_name, 'w', newline='') as write_obj:
        csv_writer = csv.writer(write_obj)
        csv_writer.writerow(["Feature_Index"])
        for idx in selected_features:
            csv_writer.writerow([idx])

def load_data_from_folder(folder_path):
    """Load CSVs from a given folder and concatenate them into one DataFrame."""
    files = glob(os.path.join(folder_path, "*.csv"))
    data_frames = [pd.read_csv(file, encoding='ISO-8859-1').fillna(0) for file in files]
    return pd.concat(data_frames, axis=0, ignore_index=True)

def convert_to_binary_labels(y):
    """Convert arousal values into binary: 0 (low arousal) or 1 (high arousal)."""
    for i in range(0, len(y)):
        if y[i] <= 0.5:
            y[i] = 0
        else:
            y[i] = 1
    return y

# Load Train and Validation Data
print("Loading training data...")
train_data = load_data_from_folder(TRAIN_DIR)
print("Loading validation data...")
val_data = load_data_from_folder(VAL_DIR)

# Prepare Features and Binary Targets
X_train = train_data.drop(columns=['LABEL_SR_Arousal'])
y_train = train_data['LABEL_SR_Arousal'].values
y_train = convert_to_binary_labels(y_train)  # Convert to binary labels

X_val = val_data.drop(columns=['LABEL_SR_Arousal'])
y_val = val_data['LABEL_SR_Arousal'].values
y_val = convert_to_binary_labels(y_val)  # Convert to binary labels

# Standardize Data
scaler_X = StandardScaler()
scaler_X.fit(X_train)

X_train_scaled = scaler_X.transform(X_train)
X_val_scaled = scaler_X.transform(X_val)

# Model Training and Evaluation
print("Training initial model...")
clf = DecisionTreeClassifier(max_depth=2, random_state=0)
clf.fit(X_train_scaled, y_train)

# Ensure directory exists for saving models
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Save initial model
joblib.dump(clf, initial_model_path)
print(f"Initial model saved to: {initial_model_path}")

# Evaluate before feature selection
y_pred_val = clf.predict(X_val_scaled)
before_f1 = f1_score(y_val, y_pred_val)

print(f"Initial F1 Score: {before_f1:.4f}")


# PSO-Based Feature Selection
def f_per_particle(m, alpha):
    """Define the cost function to evaluate classification performance using F1 Score."""
    total_features = X_train_scaled.shape[1]
    selected_features = m > 0.5
    if not np.any(selected_features):
        return np.inf  # Avoid empty feature selection

    X_train_subset = X_train_scaled[:, selected_features]
    clf.fit(X_train_subset, y_train)

    X_val_subset = X_val_scaled[:, selected_features]
    y_pred_val = clf.predict(X_val_subset)

    # Compute F1 score
    score = f1_score(y_val, y_pred_val)

    # Penalize for having fewer features
    j = (alpha * (1.0 - score) + (1.0 - alpha) * (1 - (X_train_subset.shape[1] / total_features)))

    return j


def f(x, alpha=0.88):
    """Optimize feature subset with respect to F1 Score."""
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)


# PSO Options
options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9, 'k': 30, 'p': 2}
dimensions = X_train_scaled.shape[1]
values_bound = (np.zeros(dimensions), np.ones(dimensions))
optimizer = ps.single.GlobalBestPSO(n_particles=30, dimensions=dimensions, options=options, bounds=values_bound)

print("Optimizing feature selection...")
cost, pos = optimizer.optimize(f, iters=10)  # Run PSO optimization

# Ensure selected features are valid
selected_features = pos > 0.5
if not np.any(selected_features):
    raise ValueError("No features selected during optimization.")

# Save features and their threshold values
with open(FEATURES_THRESHOLD_FILE, 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["Feature", "Threshold Value"])
    for i, threshold in enumerate(pos):
        csv_writer.writerow([i, threshold])

save_selected_features(SELECTED_FEATURES_FILE, np.where(selected_features)[0])

# Train with Selected Features
X_train_selected = X_train_scaled[:, selected_features]
X_val_selected = X_val_scaled[:, selected_features]

# Save model after feature selection
clf.fit(X_train_selected, y_train)
joblib.dump(clf, selected_model_path)
print(f"Model after feature selection saved to: {selected_model_path}")

# Evaluate with selected features
y_pred_val_selected = clf.predict(X_val_selected)
after_f1 = f1_score(y_val, y_pred_val_selected)

print(f"After Feature Selection - F1 Score: {after_f1:.4f}")

# Save Results
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, 'w', newline='') as f:
        header = ['Before_F1', 'After_F1']
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)

append_list_as_row(RESULTS_FILE, [before_f1, after_f1])

print("Results saved to", RESULTS_FILE)
print("Selected features saved to", SELECTED_FEATURES_FILE)
print("Features and their threshold values saved to", FEATURES_THRESHOLD_FILE)
