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
import pandas as pd
import numpy as np
import time
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import ParameterGrid, cross_val_score
from sklearn.metrics import mean_squared_error
import csv

# Set the property name
property_name = "{property_name}"

# Load the dataset for {description}
RDS = True
if RDS:
    descriptors_df = pd.read_csv(f'/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/{property_name}_Descriptors.csv')
    test_df = pd.read_csv(f'/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/{property_name}_Test_Descriptors.csv')
else:
    descriptors_df = pd.read_csv(f'Datasets/{property_name}_Descriptors.csv')
    test_df = pd.read_csv(f'Datasets/{property_name}_Test_Descriptors.csv')

# Function to convert SMILES to molecular descriptors
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(len(Descriptors._descList))  # Handle invalid SMILES
    return np.array([desc(mol) for name, desc in Descriptors._descList])

# Apply the conversion to the dataset
descriptors_df['descriptors'] = descriptors_df['SMILES'].apply(smiles_to_features)
descriptors_df = descriptors_df.dropna(subset=['descriptors'])  # Drop rows with invalid SMILES

# Split descriptors and target variable
X = np.array(descriptors_df['descriptors'].tolist())
y = descriptors_df['{property_name}'].values

# Define the KNN model and hyperparameters
knn = KNeighborsRegressor()
param_grid = {{
    'n_neighbors': [3, 5, 7, 9, 15, 21],
    'weights': ['uniform', 'distance'],
    'p': [1, 2, 3],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10, 20, 30, 40, 50, 100, 200, 500]
}}

# Custom Grid Search function with 5-fold cross-validation and timing
def custom_grid_search(X, y, model, param_grid, cv=5):
    results = []
    param_list = list(ParameterGrid(param_grid))
    
    with open(f'grid_search_results_{property_name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['params', 'mean_test_score', 'std_test_score', 'fit_time'])

        for params in param_list:
            model.set_params(**params)
            start_time = time.time()
            scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
            fit_time = time.time() - start_time
            mean_score = -np.mean(scores)
            std_score = np.std(scores)
            results.append({{
                'params': params,
                'mean_test_score': mean_score,
                'std_test_score': std_score,
                'fit_time': fit_time
            }})
            writer.writerow([params, mean_score, std_score, fit_time])
            print(f"Trained with params: {{params}} | Mean Test Score: {{mean_score:.4f}} | Std Test Score: {{std_score:.4f}} | Fit Time: {{fit_time:.4f}}s")
    
    return results

# Perform the custom grid search
results = custom_grid_search(X, y, knn, param_grid, cv=5)
"""

# Generate individual scripts
output_dir = "KNN_Training_Scripts"
os.makedirs(output_dir, exist_ok=True)

for property_name, description in properties:
    script_path = os.path.join(output_dir, f"train_{property_name}.py")
    with open(script_path, "w") as f:
        f.write(script_template.format(property_name=property_name, description=description))
    print(f"Generated script: {script_path}")
