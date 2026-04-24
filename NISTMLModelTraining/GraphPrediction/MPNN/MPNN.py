import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
import torch
import torch.nn.functional as F
from torch.nn import Linear
from sklearn.model_selection import KFold
import itertools
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

# Load the uploaded datasets
train_df = pd.read_csv('/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/LandoltNISTDatasets/Thermal_Conductivity_40C_Train_Descriptors.csv')
test_df = pd.read_csv('/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/LandoltNISTDatasets/Thermal_Conductivity_40C_Test_Descriptors.csv')

# Define the column names
smiles_column = 'SMILES'  # Replace with actual SMILES column name if different
target_column = 'Thermal_Conductivity_40C'  # Replace with actual target column name

# Function to convert SMILES to molecular graph
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    
    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom.GetAtomicNum())
    
    # Get bonds
    edge_index = []
    for bond in mol.GetBonds():
        edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
    
    # Convert to tensor
    x = torch.tensor(atom_features, dtype=torch.float).view(-1, 1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)

# Convert SMILES strings to graph data for training and testing
train_data_list = []
for i, row in train_df.iterrows():
    graph = smiles_to_graph(row[smiles_column])
    graph.y = torch.tensor([[row[target_column]]], dtype=torch.float)  # Ensure target has shape (N, 1)
    train_data_list.append(graph)

test_data_list = []
for i, row in test_df.iterrows():
    graph = smiles_to_graph(row[smiles_column])
    graph.y = torch.tensor([[row[target_column]]], dtype=torch.float)  # Ensure target has shape (N, 1)
    test_data_list.append(graph)

# Define the MPNN model with residual connections and dropout
class MPNN(MessagePassing):
    def __init__(self, hidden_dim, num_layers, dropout, activation):
        super(MPNN, self).__init__(aggr='add')  # "Add" aggregation
        self.convs = torch.nn.ModuleList()
        self.convs.append(Linear(1, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(Linear(hidden_dim, hidden_dim))
        self.fc1 = Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = Linear(hidden_dim // 2, 1)
        self.dropout = dropout
        self.activation = activation
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x_res = x
            x = self.activation(conv(x))
            x = self.propagate(edge_index, x=x)
            x = F.dropout(x, p=self.dropout, training=self.training) + x_res  # Residual connection
        x = global_mean_pool(x, data.batch)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def message(self, x_j):
        return x_j

# Hyperparameter grid
hidden_dims = [16, 32, 64, 128, 256]
num_layers = [1, 2, 3, 4, 5]
learning_rates = [0.01, 0.001, 0.0001]
batch_sizes = [8, 16, 32, 64, 128]
dropout_rates = [0.0]
activations = [F.relu, F.leaky_relu, torch.sigmoid, F.elu]
optimizers = ['adam', 'rmsprop' 'adagrad']

# Initialize results file
results_file = 'mpnn_thermal_conductivity_results.csv'
if not os.path.exists(results_file):
    results_df = pd.DataFrame(columns=['hidden_dim', 'num_layers', 'learning_rate', 'batch_size', 'dropout', 'activation', 'optimizer', 'avg_test_loss', 'training_time'])
    results_df.to_csv(results_file, index=False)

# Training and evaluation process
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for hidden_dim, num_layer, lr, batch_size, dropout, activation, opt in itertools.product(hidden_dims, num_layers, learning_rates, batch_sizes, dropout_rates, activations, optimizers):
    fold_results = []
    fold_times = []
    train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False)

    model = MPNN(hidden_dim, num_layer, dropout, activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) if opt == 'adam' else torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6)
    criterion = torch.nn.MSELoss()

    def train():
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
        scheduler.step(loss)

    def test(loader):
        model.eval()
        error = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out = model(data)
                error += criterion(out, data.y).item()
        return error / len(loader)

    early_stopping_patience = 10
    best_loss = float('inf')
    patience_counter = 0
    start_time = time.time()

    for epoch in tqdm(range(100), desc=f'Training for HD: {hidden_dim}, NL: {num_layer}, LR: {lr}, BS: {batch_size}, DR: {dropout}, ACT: {activation.__name__}, OPT: {opt}'):
        train()
        test_loss = test(test_loader)
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping due to no improvement in validation loss.")
            break

    end_time = time.time()
    training_time = end_time - start_time
    test_loss = test(test_loader)
    fold_results.append(test_loss)
    fold_times.append(training_time)

    avg_test_loss = np.mean(fold_results)
    avg_training_time = np.mean(fold_times)

    result = {
        'hidden_dim': hidden_dim,
        'num_layers': num_layer,
        'learning_rate': lr,
        'batch_size': batch_size,
        'dropout': dropout,
        'activation': activation.__name__,
        'optimizer': opt,
        'avg_test_loss': avg_test_loss,
        'training_time': avg_training_time
    }

    results_df = pd.DataFrame([result])
    results_df.to_csv(results_file, mode='a', header=False, index=False)
    print(f'HD: {hidden_dim}, NL: {num_layer}, LR: {lr}, BS: {batch_size}, DR: {dropout}, ACT: {activation.__name__}, OPT: {opt}, Loss: {avg_test_loss}, Time: {avg_training_time}')
