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
from itertools import product
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset for {description}
dataset = pd.read_csv("/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/LandoltNISTDatasets/{property_name}_Train.csv")
test_dataset = pd.read_csv("/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/LandoltNISTDatasets/{property_name}_Test.csv")

# Tokenize SMILES strings
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(dataset['SMILES'])
max_sequence_length = max(len(seq) for seq in tokenizer.texts_to_sequences(dataset['SMILES']))

# Prepare data
X_train = pad_sequences(tokenizer.texts_to_sequences(dataset['SMILES']), maxlen=max_sequence_length, padding='post')
y_train = dataset['{property_name}'].values
X_test = pad_sequences(tokenizer.texts_to_sequences(test_dataset['SMILES']), maxlen=max_sequence_length, padding='post')
y_test = test_dataset['{property_name}'].values

# Define the hyperparameter grid
param_grid = {{
    'optimizer': ['adam', 'rmsprop'],
    'dense_units': [16, 32, 64, 128],
    'num_layers': [1, 2, 3, 4, 5]
}}

training_params = {{
    'epochs': [100],
    'batch_size': [8, 16, 32, 64]
}}

data_sizes = [0.2, 0.4]

# Function to create the model
def create_model(optimizer='adam', dense_units=32, num_layers=1):
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_sequence_length))
    model.add(Flatten())
    for _ in range(num_layers):
        model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Perform manual grid search with cross-validation
def manual_grid_search(model_params_grid, training_params_grid, X, y, data_sizes, k=5):
    model_keys, model_values = zip(*model_params_grid.items())
    train_keys, train_values = zip(*training_params_grid.items())
    best_score = float('inf')
    best_params = None
    results = []

    for data_size in data_sizes:
        # Select a subset of the data
        X_subset, _, y_subset, _ = train_test_split(X, y, train_size=data_size, random_state=42)
        print(f"Training with data size: {{data_size * 100}}%")
        
        for model_v, train_v in product(product(*model_values), product(*train_values)):
            model_params = dict(zip(model_keys, model_v))
            train_params = dict(zip(train_keys, train_v))
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            val_scores = []

            print(f"Training model with parameters: {{model_params}} and training parameters: {{train_params}}")

            for train_index, val_index in kf.split(X_subset):
                X_train_split, X_val = X_subset[train_index], X_subset[val_index]
                y_train_split, y_val = y_subset[train_index], y_subset[val_index]

                model = create_model(**model_params)
                model.fit(X_train_split, y_train_split, epochs=train_params['epochs'], batch_size=train_params['batch_size'],
                          validation_data=(X_val, y_val),
                          callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1)

                y_val_pred = model.predict(X_val)
                val_score = mean_squared_error(y_val, y_val_pred)
                val_scores.append(val_score)

            avg_val_score = np.mean(val_scores)
            print(f"Model Params: {{model_params}}, Train Params: {{train_params}}, Avg. Validation MSE: {{avg_val_score}}")

            # Save the results
            result = {{**model_params, **train_params, 'data_size': data_size, 'avg_val_score': avg_val_score}}
            results.append(result)

            if avg_val_score < best_score:
                best_score = avg_val_score
                best_params = result

            # Save results to CSV
            results_df = pd.DataFrame(results)
            results_df.to_csv('model_performance_{property_name}.csv', index=False)

    return best_params

# Perform grid search
best_params = manual_grid_search(param_grid, training_params, X_train, y_train, data_sizes)
print(f"Best hyperparameters for {description}: {{best_params}}")

# Train final model with best hyperparameters
best_model = create_model(**{{k: v for k, v in best_params.items() if k in param_grid}})
history = best_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'])

# Save the best model
best_model.save(f"best_model_{property_name}.h5")

# Evaluate the model on test data
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE for {description}: {{mse}}")
"""

# Generate individual scripts
output_dir = "Training_Scripts"
os.makedirs(output_dir, exist_ok=True)

for property_name, description in properties:
    script_path = os.path.join(output_dir, f"train_{property_name}.py")
    with open(script_path, "w") as f:
        f.write(script_template.format(property_name=property_name, description=description))
    print(f"Generated script: {script_path}")
