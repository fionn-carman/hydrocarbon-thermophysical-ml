# Hydrocarbon thermophysical ML

Code, model weights, and a sample of the data supporting:

> Ogbomo, E.; Carman, F.; Dini, D.; Ewen, J. P. Benchmarking Molecular
> Dynamics Simulations and Machine Learning Methods for Predicting
> Hydrocarbon Thermophysical Properties. 2026.

The experimental property values used for training came from the NIST
Chemistry WebBook and the Landolt-Börnstein series. Both are licensed
and the numerical values are not redistributed here in full. Instead,
`PropertyPrediction/Data/dataset_molecules.csv` lists the 1,114 SMILES
we used along with the IUPAC name, which properties were retrieved at
40 °C and 100 °C, and the source each value came from, so anyone with
subscription access can rebuild the dataset they need. Alongside it,
`PropertyPrediction/Data/` includes the test split for each property
(SMILES and experimental value) for density, thermal conductivity, and
viscosity at 40 °C and 100 °C, as a representative sample of the data.

`PropertyPrediction/Scripts/` contains the training scripts we used for
each model family in the paper:

- `GradientBoosting/`: XGBoost
- `RandomForest/`: random forest regressors
- `KNN/`: k-nearest neighbours
- `SVM/`: support vector machines
- `MLP/`: multilayer perceptron
- `CNN/`: 1D CNN on SMILES
- `Transformer/`: transformer on SMILES
- `GraphPrediction/MPNN/`: message-passing neural network on molecular
  graphs

Each model folder has per-property training scripts (`train_*.py`) and
a small `FileGenerator.py` that generates those scripts. The scripts expect
pre-computed descriptor/fingerprint/embedding files (paths are set at
the top of each script). 

`PropertyPrediction/Models/` contains the pretrained best XGBoost models
(`best_xgboost_model_*.joblib`) for density and thermal conductivity at
40 °C and 100 °C and for viscosity at 40 °C, each with the StandardScaler
fitted during training (`scaler_*.joblib`). `verify.py` reproduces the
reported test-set accuracy: it reads each test set, regenerates the 131
RDKit descriptors from the SMILES, applies the matching scaler, runs the
model, and prints R² and RMSE. 

`LAMMPS/` contains the LAMMPS input files used for the squalane
EMD simulations: the L-OPLS data file (`Squalane_LOPLS_fixed.data`),
the init and settings includes (`Squalane_OPLS.in.init`,
`Squalane_OPLS.in.settings`), and the equilibrium MD driver
(`Squalane_OPLS_EMD.lammps`).
