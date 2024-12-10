# -*- coding: utf-8 -*-
"""
Created on Fri Aug 5 22:41:51 2022
@author: harisushehu
"""

#import libraries
import os
import csv
import joblib
import numpy as np
import pandas as pd
from math import sqrt
import pyswarms as ps
from glob import glob
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# Constants
TRAIN_DIR = "./data/train"
VAL_DIR = "./data/val"
RESULTS_FILE = "./results/regressionHR_DT.csv"
SELECTED_FEATURES_FILE = "./results/regressionHR_selected_features.csv"
FEATURES_THRESHOLD_FILE = "./results/regressionHR_features.csv"
MODEL_SAVE_PATH = "./models"
initial_model_path = os.path.join(MODEL_SAVE_PATH, "regressionHR_DT_initial.pkl")
selected_model_path = os.path.join(MODEL_SAVE_PATH, "regressionHR_DT_selected_features.pkl")

# Helper Functions
def nrmse(rmse, y_test):
    return rmse / (np.max(y_test) - np.min(y_test))

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
    files = glob(os.path.join(folder_path, "*.csv"))
    data_frames = [pd.read_csv(file, encoding='ISO-8859-1').fillna(0) for file in files]
    return pd.concat(data_frames, axis=0, ignore_index=True)

# Load Train and Validation Data
print("Loading training data...")
train_data = load_data_from_folder(TRAIN_DIR)
print("Loading validation data...")
val_data = load_data_from_folder(VAL_DIR)

# Prepare Features and Targets
X_train = train_data.drop(columns=['heartrate_mean'])
y_train = train_data['heartrate_mean'].values

X_val = val_data.drop(columns=['heartrate_mean'])
y_val = val_data['heartrate_mean'].values

# Standardize Data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))

X_val_scaled = scaler_X.transform(X_val)
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))

# Model Training and Evaluation
print("Training initial model...")
reg = DecisionTreeRegressor(max_depth=2, random_state=0)
reg.fit(X_train_scaled, y_train_scaled)

# Ensure the directory for saving models exists
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Save initial model
joblib.dump(reg, initial_model_path)
print(f"Initial model saved to: {initial_model_path}")

# Evaluate Before Feature Selection
y_pred_val = reg.predict(X_val_scaled)
before_rmse = sqrt(mean_squared_error(y_val_scaled, y_pred_val))
before_nrmse = nrmse(before_rmse, y_val_scaled.flatten())

print(f"Initial RMSE: {before_rmse:.4f}, NRMSE: {before_nrmse:.4f}")

# PSO-Based Feature Selection
def f_per_particle(m, alpha):
    total_features = X_train_scaled.shape[1]
    selected_features = m > 0.5
    if not np.any(selected_features):
        return np.inf  # Avoid empty feature selection

    X_train_subset = X_train_scaled[:, selected_features]
    reg.fit(X_train_subset, y_train_scaled)

    X_val_subset = X_val_scaled[:, selected_features]
    y_pred_val = reg.predict(X_val_subset)

    P = r2_score(y_val_scaled, y_pred_val)

    j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_train_subset.shape[1] / total_features)))

    return j

def f(x, alpha=0.88):
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)

options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9, 'k': 30, 'p': 2}

# Use GlobalBestPSO
dimensions = X_train_scaled.shape[1]
values_bound = (np.zeros(dimensions), np.ones(dimensions))
optimizer = ps.single.GlobalBestPSO(n_particles=30, dimensions=dimensions, options=options, bounds=values_bound)

print("Optimizing feature selection...")
cost, pos = optimizer.optimize(f, iters=2) #100

# Ensure selected features are valid
selected_features = pos > 0.5
if not np.any(selected_features):
    raise ValueError("No features selected during optimization.")
    
# Save features and their threshold values
with open(FEATURES_THRESHOLD_FILE, 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["Feature", "Threshold Value"])  # Write header
    for i, threshold in enumerate(pos):
        csv_writer.writerow([i, threshold])
    
# Save selected feature indices to CSV
save_selected_features(SELECTED_FEATURES_FILE, np.where(selected_features)[0])

# Train with Selected Features
X_train_selected = X_train_scaled[:, selected_features]
X_val_selected = X_val_scaled[:, selected_features]

# Save model after feature selection
joblib.dump(reg, selected_model_path)
print(f"Model after feature selection saved to: {selected_model_path}")

# Make prediction with Selected Features
reg.fit(X_train_selected, y_train_scaled)
y_pred_val_selected = reg.predict(X_val_selected)

# Evaluate After Feature Selection
after_rmse = sqrt(mean_squared_error(y_val_scaled, y_pred_val_selected))
after_nrmse = nrmse(after_rmse, y_val_scaled.flatten())

print(f"After Feature Selection - RMSE: {after_rmse:.4f}, NRMSE: {after_nrmse:.4f}")

# Save Results
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, 'w', newline='') as f:
        header = ['Before_RMSE', 'Before_NRMSE', 'After_RMSE', 'After_NRMSE']
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)

append_list_as_row(RESULTS_FILE, [before_rmse, before_nrmse, after_rmse, after_nrmse])

print("Results saved to", RESULTS_FILE)
print("Selected features saved to", SELECTED_FEATURES_FILE)
print("Features and their threshold values saved to", FEATURES_THRESHOLD_FILE)

