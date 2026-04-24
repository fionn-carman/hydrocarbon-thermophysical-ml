import os

# Define the base script template
script_template = """
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
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

# Load the dataset
train_df = pd.read_csv('{train_path}')
test_df = pd.read_csv('{test_path}')

# Define the column names
smiles_column = 'SMILES'
target_column = '{target_column}'

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

# Define the MPNN model
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

# Hyperparameters
hidden_dim = {hidden_dim}
num_layers = {num_layer}
learning_rates = [0.01, 0.001, 0.0001]
batch_sizes = [8, 16, 32]
dropout_rates = [0.0, 0.2]
activations = [F.relu, F.leaky_relu]
optimizers = ['adam', 'rmsprop']

# Training and evaluation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results_file = 'results.csv'

if not os.path.exists(results_file):
    pd.DataFrame(columns=['hidden_dim', 'num_layers', 'learning_rate', 'batch_size', 'dropout', 'activation', 'optimizer', 'avg_test_loss', 'training_time']).to_csv(results_file, index=False)

for lr in learning_rates:
    for batch_size in batch_sizes:
        for dropout in dropout_rates:
            for activation in activations:
                for opt in optimizers:
                    model = MPNN(hidden_dim, num_layers, dropout, activation).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr) if opt == 'adam' else \
                                torch.optim.RMSprop(model.parameters(), lr=lr)
                    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6)
                    criterion = torch.nn.MSELoss()
                    
                    train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
                    test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False)
                    
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

                    best_loss = float('inf')
                    for epoch in tqdm(range(100)):
                        train()
                        test_loss = test(test_loader)
                        if test_loss < best_loss:
                            best_loss = test_loss
                    
                    with open(results_file, 'a') as f:
                        f.write(f"{hidden_dim},{{num_layers}},{{lr}},{{batch_size}},{{dropout}},{{activation.__name__}},{{opt}},{{best_loss}},-\\n")
"""

# Properties and paths
properties = [
    "Thermal_Conductivity_40C", "Thermal_Conductivity_100C",
    "Density_40C", "Density_100C",
    "Viscosity_40C", "Viscosity_100C",
    "Heat_Capacity_40C", "Heat_Capacity_100C"
]

# Hyperparameter values
hidden_dims = [16, 32, 64, 128]
num_layers = [1, 2, 3, 4, 5]

# Create directory structure and generate scripts
base_path = 'generated_scripts'
os.makedirs(base_path, exist_ok=True)

for prop in properties:
    prop_dir = os.path.join(base_path, prop)
    os.makedirs(prop_dir, exist_ok=True)
    
    train_path = f'/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/LandoltNISTDatasets/{prop}_Train_Descriptors.csv'
    test_path = f'/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/LandoltNISTDatasets/{prop}_Test_Descriptors.csv'
    
    for hidden_dim in hidden_dims:
        for num_layer in num_layers:
            script_content = script_template.format(
                train_path=train_path,
                test_path=test_path,
                target_column=prop,
                hidden_dim=hidden_dim,
                num_layer=num_layer
            )
            script_filename = os.path.join(prop_dir, f'train_hd_{hidden_dim}_nl_{num_layer}.py')
            with open(script_filename, 'w') as script_file:
                script_file.write(script_content)
