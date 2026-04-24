import os

# Define properties and corresponding filenames
properties = [
    ("Thermal_Conductivity_40C", "Thermal Conductivity at 40Â°C", "Thermal_Conductivity_40C"),
    ("Thermal_Conductivity_100C", "Thermal Conductivity at 100Â°C", "Thermal_Conductivity_100C"),
    ("Density_40C", "Density at 40Â°C", "Density_40C"), 
    ("Density_100C", "Density at 100Â°C", "Density_100C"),
    ("Viscosity_40C", "Viscosity at 40Â°C", "Viscosity_40C"),
    ("Viscosity_100C", "Viscosity at 100Â°C", "Viscosity_100C"),
    ("Heat_Capacity_40C", "Heat Capacity at 40Â°C", "Heat_Capacity_40C"),
    ("Heat_Capacity_100C", "Heat Capacity at 100Â°C", "Heat_Capacity_100C")
]

# Template for the training script
script_template = """
import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
import matplotlib.pyplot as plt
import xgboost as xgb
import os

# Load dataset for {description}
RDS = True
if RDS:
    descriptors_df = pd.read_csv('/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/LandoltNISTDatasets/{property_name}_Train_Descriptors.csv')
    test_df = pd.read_csv('/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/LandoltNISTDatasets/{property_name}_Test_Descriptors.csv')
else:
    descriptors_df = pd.read_csv('Datasets/{property_name}_Descriptors.csv')
    test_df = pd.read_csv('Datasets/{property_name}_Test_Descriptors.csv')

# Separate features and target variable
X = descriptors_df.drop(columns=['{column_name}', 'SMILES'])
y = descriptors_df['{column_name}']
X_test = test_df.drop(columns=['{column_name}', 'SMILES'])
y_test = test_df['{column_name}']

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

param_grid = {{
    'n_estimators': [100, 200, 300, 500, 1000, 2000, 5000],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5],
    'max_depth': [3, 4, 5, 6, 10, 20, 50, 100],
    'tree_method': ['hist'],  # Use GPU
}}


# Custom grid search function
def custom_grid_search(X_train, y_train, X_test, y_test, param_grid):
    results = []
    for n_estimators in param_grid['n_estimators']:
        for learning_rate in param_grid['learning_rate']:
            for max_depth in param_grid['max_depth']:
                params = {{
                    'n_estimators': n_estimators,
                    'learning_rate': learning_rate,
                    'max_depth': max_depth,
                    'tree_method': 'hist',
                    'use_label_encoder': False,
                    'eval_metric': 'rmse'
                }}
                model = xgb.XGBRegressor(**params)
                
                # Measure training time
                start_time = time.time()
                model.fit(X_train, y_train)
                end_time = time.time()
                train_time = end_time - start_time

                # Evaluate on test set
                y_test_pred = model.predict(X_test)
                test_mse = mean_squared_error(y_test, y_test_pred)
                test_r2 = r2_score(y_test, y_test_pred)

                result = {{
                    'Params': params,
                    'Test MSE': test_mse,
                    'Test R2': test_r2,
                    'Train Time': train_time
                }}
                results.append(result)

                # Save results to a CSV file in append mode
                results_df = pd.DataFrame([result])
                results_df.to_csv('xgboost_{property_name}_training_results.csv', mode='a', header=not os.path.isfile('xgboost_{property_name}_training_results.csv'), index=False)
                
                print(f"Params: {{params}}, Test MSE: {{test_mse}}, Test R2: {{test_r2}}")

    return results

# Perform custom grid search
results = custom_grid_search(X_train_scaled, y_train, X_test_scaled, y_test, param_grid)

# Find the best model based on test MSE
best_result = min(results, key=lambda x: x['Test MSE'])
best_params = best_result['Params']

# Train the best model on the full training data
best_model = xgb.XGBRegressor(**best_params)
best_model.fit(X_train_scaled, y_train)

# Save the best model
dump(best_model, 'best_xgboost_model_{property_name}.joblib')

# Evaluate the best model on the test set
y_test_pred = best_model.predict(X_test_scaled)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
print(f"Best Params: {{best_params}}, Test MSE: {{test_mse}}, Test R2: {{test_r2}}")

# Feature importance
feature_importance = best_model.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(10, 6))
plt.barh(range(X.shape[1]), feature_importance[sorted_idx], align='center')
plt.yticks(range(X.shape[1]), np.array(X.columns)[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance for {description}')
plt.savefig('feature_importance_{property_name}.png')
plt.show()
"""

# Generate individual scripts
output_dir = "XGBoost_Training_Scripts"
os.makedirs(output_dir, exist_ok=True)

for property_name, description, column_name in properties:
    script_path = os.path.join(output_dir, f"train_{property_name}.py")
    with open(script_path, "w") as f:
        f.write(script_template.format(property_name=property_name, description=description, column_name=column_name))
    print(f"Generated script: {script_path}")
