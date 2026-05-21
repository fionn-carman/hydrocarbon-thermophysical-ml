
import pandas as pd
import numpy as np
import time
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import ParameterGrid, cross_val_score
from sklearn.metrics import mean_squared_error
import csv

# Load the dataset for Density at 100°C
file_path = "Datasets/Density_100C_Dataset.csv"
data = pd.read_csv(file_path)

# Function to convert SMILES to molecular descriptors
def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        descriptors = np.array([
            Descriptors.MolWt(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.MolLogP(mol)
        ])
        return descriptors
    else:
        return np.array([np.nan]*5)

# Apply the conversion to the dataset
data['descriptors'] = data['SMILES'].apply(smiles_to_descriptors)
data = data.dropna(subset=['descriptors'])  # Drop rows with invalid SMILES

# Split descriptors and target variable
X = np.array(data['descriptors'].tolist())
y = data['Density_100C'].values

# Define the KNN model and hyperparameters
knn = KNeighborsRegressor()
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 15, 21],
    'weights': ['uniform', 'distance'],
    'p': [1, 2, 3],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10, 20, 30, 40, 50, 100, 200, 500]
}

# Custom Grid Search function with 5-fold cross-validation and timing
def custom_grid_search(X, y, model, param_grid, cv=5):
    results = []
    param_list = list(ParameterGrid(param_grid))
    
    with open('grid_search_results_Density_100C.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['params', 'mean_test_score', 'std_test_score', 'fit_time'])

        for params in param_list:
            model.set_params(**params)
            start_time = time.time()
            scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
            fit_time = time.time() - start_time
            mean_score = -np.mean(scores)
            std_score = np.std(scores)
            results.append({
                'params': params,
                'mean_test_score': mean_score,
                'std_test_score': std_score,
                'fit_time': fit_time
            })
            writer.writerow([params, mean_score, std_score, fit_time])
            print(f"Trained with params: {params} | Mean Test Score: {mean_score:.4f} | Std Test Score: {std_score:.4f} | Fit Time: {fit_time:.4f}s")
    
    return results

# Perform the custom grid search
results = custom_grid_search(X, y, knn, param_grid, cv=5)
