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
script_template = """import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, LayerNormalization, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import csv
import os

# Load dataset for {description}
train_dataset = pd.read_csv("/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/LandoltNISTDatasets/{property_name}_Train.csv")
test_dataset = pd.read_csv("/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/LandoltNISTDatasets/{property_name}_Test.csv")

# Tokenize SMILES strings
tokenizer = Tokenizer(char_level=True)  # Tokenize at character level
tokenizer.fit_on_texts(train_dataset['SMILES'])
sequences_train = tokenizer.texts_to_sequences(train_dataset['SMILES'])
sequences_test = tokenizer.texts_to_sequences(test_dataset['SMILES'])
max_sequence_length = max(len(seq) for seq in sequences_train)

# Pad sequences to ensure uniform input length
X_train = pad_sequences(sequences_train, maxlen=max_sequence_length, padding='post')
X_test = pad_sequences(sequences_test, maxlen=max_sequence_length, padding='post')
y_train = train_dataset['{property_name}'].values
y_test = test_dataset['{property_name}'].values

# Define the hyperparameter grid
param_grid = {{
    'head_size': [8, 16, 32, 64],
    'num_heads': [5, 10],
    'ff_dim': [32, 64, 128],
    'dropout': [0.0],
    'num_transformer_blocks': [1, 2, 3],
    'dense_units': [16, 32],
    'optimizer': ['adam', 'rmsprop']
}}

training_params = {{
    'epochs': [100],
    'batch_size': [16, 32, 64]
}}

# Function to create the model
def create_model(head_size, num_heads, ff_dim, dropout, num_transformer_blocks, dense_units, optimizer):
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        # Normalization and Attention
        x = LayerNormalization(epsilon=1e-6)(inputs)
        x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        x = Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = LayerNormalization(epsilon=1e-6)(res)
        x = Dense(ff_dim, activation="relu")(x)
        x = Dropout(dropout)(x)
        x = Dense(inputs.shape[-1])(x)
        return x + res

    inputs = Input(shape=(max_sequence_length,))
    x = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64)(inputs)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = Flatten()(x)
    x = Dense(dense_units, activation='relu')(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Perform manual grid search
def manual_grid_search(model_params_grid, training_params_grid, X_train, y_train, X_test, y_test):
    model_keys, model_values = zip(*model_params_grid.items())
    train_keys, train_values = zip(*training_params_grid.items())
    best_score = float('inf')
    best_params = None

    # Open a CSV file to save the performance metrics
    with open(f'results_{property_name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model Params', 'Train Params', 'Validation MSE', 'R2 Score'])

        for model_v, train_v in product(product(*model_values), product(*train_values)):
            model_params = dict(zip(model_keys, model_v))
            train_params = dict(zip(train_keys, train_v))

            model = create_model(**model_params)
            model.fit(X_train, y_train, epochs=train_params['epochs'], batch_size=train_params['batch_size'],
                      validation_data=(X_test, y_test),
                      callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=0)

            y_test_pred = model.predict(X_test)
            val_score = mean_squared_error(y_test, y_test_pred)
            r2 = r2_score(y_test, y_test_pred)

            # Save the performance metrics to the CSV
            writer.writerow([model_params, train_params, val_score, r2])

            # Save the best model
            if val_score < best_score:
                best_score = val_score
                best_params = {{**model_params, **train_params}}
                model.save(f'best_transformer_model_{property_name}.h5')

    return best_params

# Perform grid search
best_params = manual_grid_search(param_grid, training_params, X_train, y_train, X_test, y_test)
print(f"Best hyperparameters: {{best_params}}")

# Load the best model
best_model = load_model(f'best_transformer_model_{property_name}.h5')

# Evaluate the best model
y_pred_test = best_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)
print(f"Test MSE: {{test_mse}}")
print(f"Test R2: {{test_r2}}")
"""

# Generate individual scripts
output_dir = "Transformer_Training_Scripts"
os.makedirs(output_dir, exist_ok=True)

for property_name, description in properties:
    script_path = os.path.join(output_dir, f"train_{property_name}.py")
    with open(script_path, "w") as f:
        f.write(script_template.format(property_name=property_name, description=description))
    print(f"Generated script: {script_path}")
