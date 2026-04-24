
import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load dataset

property_name = "Thermal_Conductivity_100C"
dataset = pd.read_csv("/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/LandoltNISTDatasets/Thermal_Conductivity_100C_Train.csv")
test_dataset = pd.read_csv("/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/LandoltNISTDatasets/Thermal_Conductivity_100C_Test.csv")

# Tokenize SMILES strings
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(dataset['SMILES'])
max_sequence_length = max(len(seq) for seq in tokenizer.texts_to_sequences(dataset['SMILES']))

# Prepare data
X_train = pad_sequences(tokenizer.texts_to_sequences(dataset['SMILES']), maxlen=max_sequence_length, padding='post')
y_train = dataset[property_name].values
X_test = pad_sequences(tokenizer.texts_to_sequences(test_dataset['SMILES']), maxlen=max_sequence_length, padding='post')
y_test = test_dataset[property_name].values

# Define the hyperparameter grid
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'filters': [32, 64, 128],
    'kernel_size': [3, 5, 7],
    'dense_units': [16, 32, 64],
    'num_layers': [1, 2, 3]
}

training_params = {
    'epochs': [100],
    'batch_size': [16, 32, 64, 128]
}

# Function to create the model
def create_model(optimizer='adam', filters=128, kernel_size=3, dense_units=32, num_layers=1):
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_sequence_length))
    for i in range(num_layers):
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
        if i < num_layers - 1:  # Add MaxPooling1D except for the last conv layer
            model.add(MaxPooling1D(pool_size=2))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Perform manual grid search with cross-validation
def manual_grid_search(model_params_grid, training_params_grid, X, y, k=5):
    model_keys, model_values = zip(*model_params_grid.items())
    train_keys, train_values = zip(*training_params_grid.items())
    best_score = float('inf')
    best_params = None

    for model_v, train_v in product(product(*model_values), product(*train_values)):
        model_params = dict(zip(model_keys, model_v))
        train_params = dict(zip(train_keys, train_v))
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        val_scores = []

        print(f"Training model with parameters: {model_params}, training parameters: {train_params}")

        for train_index, val_index in kf.split(X):
            X_train_split, X_val = X[train_index], X[val_index]
            y_train_split, y_val = y[train_index], y[val_index]

            model = create_model(**model_params)
            model.fit(X_train_split, y_train_split, epochs=train_params['epochs'], batch_size=train_params['batch_size'],
                      validation_data=(X_val, y_val),
                      callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1)

            y_val_pred = model.predict(X_val)
            val_score = mean_squared_error(y_val, y_val_pred)
            val_scores.append(val_score)

        avg_val_score = np.mean(val_scores)
        print(f"Model Params: {model_params}, Train Params: {train_params}, Avg. Validation MSE: {avg_val_score}")

        # Save each model's performance to a CSV file
        result = {
            'Train Size': size,
            'Model Params': model_params,
            'Train Params': train_params,
            'Validation MSE': avg_val_score
        }

        results_df = pd.DataFrame([result])
        results_df.to_csv('cnn_model_training_results.csv', mode='a', header=not header_written, index=False)
        header_written = True

        if avg_val_score < best_score:
            best_score = avg_val_score
            best_params = {**model_params, **train_params}


    if avg_val_score < best_score:
        best_score = avg_val_score
        best_params = {**model_params, **train_params}

    return best_params

# Perform grid search
best_params = manual_grid_search(param_grid, training_params, X_train, y_train)
print(f"Best hyperparameters: {best_params}")

# Train final model with best hyperparameters
best_model = create_model(**{k: v for k, v in best_params.items() if k in param_grid})
history = best_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'])

# Save the best model
best_model.save(f"best_cnn_model_Thermal_Conductivity_100C.h5")

# Evaluate the model on test data
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE for Thermal_Conductivity_100C: {mse}")
