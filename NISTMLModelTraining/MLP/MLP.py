import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, Masking, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os
from os.path import join

RDS = True
CWD = os.getcwd()

# Load dataset

if RDS:
    Dataset = pd.read_csv('/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/FinalDataset.csv')
else:
    Dataset = pd.read_csv('Datasets/FinalDataset.csv')

# Tokenize SMILES strings
tokenizer = Tokenizer(char_level=True)  # Tokenize at character level
tokenizer.fit_on_texts(Dataset['smiles'])
sequences = tokenizer.texts_to_sequences(Dataset['smiles'])
max_sequence_length = max(len(seq) for seq in sequences)

# Pad sequences to ensure uniform input length
X = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
y = Dataset['visco@40C[cP]'].values

# Define the hyperparameter grid
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'dense_units': [16, 32, 64, 128],
    'num_layers': [1, 2, 3, 4, 5]
}

training_params = {
    'epochs': [100],
    'batch_size': [8, 16, 32, 64]
}

# Different data sizes to be tested
data_sizes = [0.2, 0.4]#, 0.6, 0.8]  # 60%, 80%, and 100% of the data

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
        print(f"Training with data size: {data_size * 100}%")
        
        for model_v, train_v in product(product(*model_values), product(*train_values)):
            model_params = dict(zip(model_keys, model_v))
            train_params = dict(zip(train_keys, train_v))
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            val_scores = []

            print(f"Training model with parameters: {model_params} and training parameters: {train_params}")

            for train_index, val_index in kf.split(X_subset):
                X_train, X_val = X_subset[train_index], X_subset[val_index]
                y_train, y_val = y_subset[train_index], y_subset[val_index]

                model = create_model(**model_params)
                model.fit(X_train, y_train, epochs=train_params['epochs'], batch_size=train_params['batch_size'],
                          validation_data=(X_val, y_val),
                          callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1)

                y_val_pred = model.predict(X_val)
                val_score = mean_squared_error(y_val, y_val_pred)
                val_scores.append(val_score)

            avg_val_score = np.mean(val_scores)
            print(f"Model Params: {model_params}, Train Params: {train_params}, Avg. Validation MSE: {avg_val_score}")

            # Save the results
            result = {**model_params, **train_params, 'data_size': data_size, 'avg_val_score': avg_val_score}
            results.append(result)

            if avg_val_score < best_score:
                best_score = avg_val_score
                best_params = result

            # Save results to CSV
            results_df = pd.DataFrame(results)
            results_df.to_csv('model_performance.csv', index=False)

    return best_params

# Perform grid search
best_params = manual_grid_search(param_grid, training_params, X, y, data_sizes)
print(f"Best hyperparameters: {best_params}")

# Train final model with best hyperparameters
# best_model = create_model(**{k: v for k, v in best_params.items() if k in param_grid})
# final_data_size = best_params['data_size']
# X_train_final, _, y_train_final, _ = train_test_split(X, y, train_size=final_data_size, random_state=42)
# history = best_model.fit(X_train_final, y_train_final, epochs=best_params['epochs'], batch_size=best_params['batch_size'])

# # Load the best model
# best_model = load_model(join(CWD, 'checkpoint.model.keras'))

# # Make predictions with the best model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# y_pred = best_model.predict(X_test)

# # Evaluate the best model
# mse = mean_squared_error(y_test, y_pred)
# print(f'Best model MSE: {mse}')

# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper right')
# plt.savefig(join(CWD))
# plt.show()
