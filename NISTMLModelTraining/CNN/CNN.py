import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Embedding, Masking
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import shap
import os

RDS = True
CWD = os.getcwd()

# Load dataset
if RDS:
    Dataset = pd.read_csv('/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/LandoltNISTDatasets/FinalDataset.csv')
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
    'filters': [32, 64, 128],
    'kernel_size': [3, 5, 7],
    'dense_units': [16, 32, 64],
    'num_layers': [1, 2, 3]
}

training_params = {
    'epochs': [100],
    'batch_size': [16, 32,64, 128]
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
def manual_grid_search(model_params_grid, training_params_grid, X, y, train_sizes, k=5):
    model_keys, model_values = zip(*model_params_grid.items())
    train_keys, train_values = zip(*training_params_grid.items())
    best_score = float('inf')
    best_params = None
    header_written = False

    for size in train_sizes:
        X_train_partial, _, y_train_partial, _ = train_test_split(X, y, train_size=size, random_state=42)
        
        for model_v, train_v in product(product(*model_values), product(*train_values)):
            model_params = dict(zip(model_keys, model_v))
            train_params = dict(zip(train_keys, train_v))
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            val_scores = []

            print(f"Training model with parameters: {model_params}, training parameters: {train_params}, and train size: {size}")

            for train_index, val_index in kf.split(X_train_partial):
                X_train, X_val = X_train_partial[train_index], X_train_partial[val_index]
                y_train, y_val = y_train_partial[train_index], y_train_partial[val_index]

                model = create_model(**model_params)
                model.fit(X_train, y_train, epochs=train_params['epochs'], batch_size=train_params['batch_size'],
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

    return best_params

# Define train sizes
train_sizes = [0.2]

# Perform grid search
best_params = manual_grid_search(param_grid, training_params, X, y, train_sizes)
print(f"Best hyperparameters: {best_params}")

# Train final model with best hyperparameters
best_model = create_model(**{k: v for k, v in best_params.items() if k in param_grid})
history = best_model.fit(X, y, epochs=best_params['epochs'], batch_size=best_params['batch_size'])

# Save the best model
best_model.save('best_cnn_model.h5')

# Load the best model
best_model = load_model('best_cnn_model.h5')

# Make predictions with the best model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = best_model.predict(X_test)

# Evaluate the best model
mse = mean_squared_error(y_test, y_pred)
print(f'Best model MSE: {mse}')

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('cnn_model_loss.png')
plt.show()

# Perform individual predictions based on input SMILES strings
def predict_smiles(smiles_string):
    sequence = tokenizer.texts_to_sequences([smiles_string])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post')
    prediction = best_model.predict(padded_sequence)
    return prediction[0]

# Example prediction
example_smiles = "CCO"
print(f'Predicted viscosity for {example_smiles}: {predict_smiles(example_smiles)}')

# SHAP values for feature importance
explainer = shap.KernelExplainer(best_model.predict, X_train[:100])  # Using a small subset for explanation
shap_values = explainer.shap_values(X_test[:10])  # Again, a small subset for the example

# Plot SHAP values
shap.summary_plot(shap_values, X_test[:10], feature_names=tokenizer.index_word)
# Save the SHAP plot
plt.savefig('shap_summary_plot.png')
