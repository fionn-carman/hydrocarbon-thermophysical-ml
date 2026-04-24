import os

# Define properties and corresponding filenames
properties = [
    ("Thermal_Conductivity_40C", "Thermal Conductivity at 40C"),
    ("Thermal_Conductivity_100C", "Thermal Conductivity at 100C"),
    ("Density_40C", "Density at 40C"),
    ("Density_100C", "Density at 100C"),
    ("Viscosity_40C", "Viscosity at 40C"),
    ("Viscosity_100C", "Viscosity at 100C"),
    ("Heat_Capacity_40C", "Heat Capacity at 40C"),
    ("Heat_Capacity_100C", "Heat Capacity at 100C")
]

# Template for the training script
script_template = """
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
import matplotlib.pyplot as plt
import joblib
import time
import csv
import os
from os.path import join

CWD = os.getcwd()

# Load dataset for {description}
RDS = True
if RDS:
    descriptors_df = pd.read_csv('/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/LandoltNISTDatasets/{property_name}_Train_Descriptors.csv')
    test_df = pd.read_csv('/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/LandoltNISTDatasets/{property_name}_Test_Descriptors.csv')
else:
    descriptors_df = pd.read_csv('Datasets/{property_name}_Descriptors.csv')
    test_df = pd.read_csv('Datasets/{property_name}_Test_Descriptors.csv')

# Remove Unnamed column
descriptors_df = descriptors_df.loc[:, ~descriptors_df.columns.str.contains('^Unnamed')]

# Separate features and target variable
X = descriptors_df.drop(columns=['{property_name}'])
y = descriptors_df['{property_name}']
X_test = test_df.drop(columns=['{property_name}'])
y_test = test_df['{property_name}']

# Define the scaler and model
scaler = StandardScaler()
model = SVR()

# Define the parameter grid for grid search
param_grid = {{
    'C': [0.01, 0.1, 1, 3, 5, 10],
    'epsilon': [0.001, 0.01, 0.1, 0.2],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}}

# Initialize KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform grid search and save results incrementally
results = []
with open(join(CWD, f'grid_search_results_{property_name}.csv'), mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Parameters', 'Mean Test Score', 'Standard Deviation Test Score', 'Rank', 'Training Time'])

    for C in param_grid['C']:
        for epsilon in param_grid['epsilon']:
            for kernel in param_grid['kernel']:
                current_params = {{'C': [C], 'epsilon': [epsilon], 'kernel': [kernel]}}
                grid_search = GridSearchCV(estimator=model, param_grid=current_params, scoring='neg_mean_squared_error', cv=kf, verbose=2)
                
                model_start_time = time.time()
                grid_search.fit(scaler.fit_transform(X), y)
                model_end_time = time.time()

                training_time = (model_end_time - model_start_time) / 5

                # Save intermediate results
                for i in range(len(grid_search.cv_results_['mean_test_score'])):
                    result = {{
                        'params': grid_search.cv_results_['params'][i],
                        'mean_test_score': -grid_search.cv_results_['mean_test_score'][i],
                        'std_test_score': grid_search.cv_results_['std_test_score'][i],
                        'rank_test_score': grid_search.cv_results_['rank_test_score'][i],
                        'training_time': training_time
                    }}
                    results.append(result)
                    writer.writerow([result['params'], result['mean_test_score'], result['std_test_score'], result['rank_test_score'], result['training_time']])
                    file.flush()  # Ensure the data is written to the file immediately

# Best model and parameters
best_model = grid_search.best_estimator_
print(f"Best parameters: {{grid_search.best_params_}}")

# Save the best model
joblib.dump(best_model, join(CWD, f'best_svr_model_{property_name}.pkl'))

# Evaluate on test set
y_pred_test = best_model.predict(scaler.transform(X_test))
test_mse = mean_squared_error(y_test, y_pred_test)
print(f"Test MSE: {{test_mse}}")
"""

# Generate individual scripts
output_dir = "SVR_Training_Scripts"
os.makedirs(output_dir, exist_ok=True)

for property_name, description in properties:
    script_path = os.path.join(output_dir, f"train_{property_name}.py")
    with open(script_path, "w") as f:
        f.write(script_template.format(property_name=property_name, description=description))
    print(f"Generated script: {script_path}")
