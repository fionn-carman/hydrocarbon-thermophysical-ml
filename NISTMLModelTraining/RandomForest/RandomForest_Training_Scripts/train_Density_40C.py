
import numpy as np
import pandas as pd
import time
from itertools import product
from sklearn.model_selection import train_test_split, KFold, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

# Load dataset for Density at 40°C
RDS = False
if RDS:
    descriptors_df = pd.read_csv('/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/Density_40C_Descriptors.csv')
    test_df = pd.read_csv('/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/Density_40C_Test_Descriptors.csv')
else:
    descriptors_df = pd.read_csv('Datasets/Density_40C_Descriptors.csv')
    test_df = pd.read_csv('Datasets/Density_40C_Test_Descriptors.csv')

# Separate features and target variable
X = descriptors_df.drop(columns=['Density_40C'])
y = descriptors_df['Density_40C']
X_test = test_df.drop(columns=['Density_40C'])
y_test = test_df['Density_40C']

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

# Perform manual grid search with cross-validation
def manual_grid_search(param_grid, X, y, k=5):
    keys, values = zip(*param_grid.items())
    best_score = float('inf')
    best_params = None
    header_written = False

    for v in product(*values):
        params = dict(zip(keys, v))
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        val_scores = []
        train_times = []

        print(f"Training model with parameters: {params}")

        for train_index, val_index in kf.split(X):
            X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

            model = RandomForestRegressor(**params, random_state=42)
            start_time = time.time()
            model.fit(X_train_fold, y_train_fold)
            end_time = time.time()
            train_time = end_time - start_time

            train_times.append(train_time)

            y_val_pred = model.predict(X_val_fold)
            val_score = mean_squared_error(y_val_fold, y_val_pred)
            val_scores.append(val_score)

        avg_val_score = np.mean(val_scores)
        avg_train_time = np.mean(train_times)
        print(f"Params: {params}, Avg. Validation MSE: {avg_val_score}, Avg. Train Time: {avg_train_time}")

        # Save each model's performance to a CSV file
        result = {
            'Params': params,
            'Validation MSE': avg_val_score,
            'Train Time': avg_train_time
        }
        results_df = pd.DataFrame([result])
        results_df.to_csv('random_forest_Density_40C_training_results.csv', mode='a', header=not os.path.isfile('random_forest_Density_40C_training_results.csv'), index=False)
        header_written = True

        if avg_val_score < best_score:
            best_score = avg_val_score
            best_params = params

    return best_params

# Perform grid search
best_params = manual_grid_search(param_grid, X, y)
print(f"Best hyperparameters: {best_params}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train final model with best hyperparameters
best_model = RandomForestRegressor(**best_params, random_state=42)
best_model.fit(X_train, y_train)

# Save the best model
joblib.dump(best_model, 'best_random_forest_model_Density_40C.pkl')

# Load the best model
loaded_model = joblib.load('best_random_forest_model_Density_40C.pkl')

# Make predictions with the best model
y_pred = loaded_model.predict(X_test)

# Evaluate the best model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

# Learning Curve
train_sizes_lc, train_scores, test_scores = learning_curve(loaded_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 5))

# Calculate mean and standard deviation
train_scores_mean = np.sqrt(-np.mean(train_scores, axis=1))
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.sqrt(-np.mean(test_scores, axis=1))
test_scores_std = np.std(test_scores, axis=1)

# Plot the learning curve
plt.figure()
plt.title("Learning Curve for RandomForestRegressor (Density at 40°C)")
plt.xlabel("Training examples")
plt.ylabel("Root Mean Squared Error")
plt.grid()

# Plot the mean train and test scores with standard deviation
plt.fill_between(train_sizes_lc, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes_lc, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes_lc, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes_lc, test_scores_mean, 'o-', color="g", label="Cross-validation score")

plt.legend(loc="best")
plt.savefig('random_forest_learning_curve_Density_40C.png')
plt.show()
