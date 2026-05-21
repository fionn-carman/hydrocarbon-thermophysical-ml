#!/usr/bin/env python3
"""
Check the pretrained XGBoost models against the held-out test sets.

For each property it reads the test SMILES and experimental values from
../Data, regenerates the 131 RDKit descriptors from the SMILES, applies the
StandardScaler that was fitted during training (saved next to each model), runs
the model, and prints R2 and RMSE on the test set.

Everything the model needs travels with it: the weights and the scaler are both
here in Models/, so no training data is required to reproduce the numbers.

    pip install numpy pandas scikit-learn xgboost joblib rdkit
    python verify.py
"""
import os
import numpy as np
import pandas as pd
from joblib import load
from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from sklearn.metrics import r2_score, mean_squared_error

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "Data")

# Properties with a pretrained model in this folder.
PROPERTIES = [
    "Density_40C",
    "Density_100C",
    "Thermal_Conductivity_40C",
    "Thermal_Conductivity_100C",
    "Viscosity_40C",
]

# The 131 RDKit descriptors, in the order the models expect them.
FEATURES = [
    'MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex', 'qed', 'SPS',
    'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'MaxPartialCharge',
    'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'FpDensityMorgan1',
    'FpDensityMorgan2', 'FpDensityMorgan3', 'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI',
    'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'AvgIpc',
    'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v',
    'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3',
    'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2',
    'PEOE_VSA3', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1',
    'SMR_VSA10', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA9', 'SlogP_VSA1',
    'SlogP_VSA11', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6',
    'SlogP_VSA8', 'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA2', 'EState_VSA3',
    'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9',
    'VSA_EState1', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6',
    'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount',
    'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles',
    'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds',
    'NumSaturatedCarbocycles', 'NumSaturatedRings', 'RingCount', 'MolLogP', 'MolMR', 'fr_Al_COO',
    'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO',
    'fr_allylic_oxid', 'fr_aryl_methyl', 'fr_benzene', 'fr_bicyclic', 'fr_ester', 'fr_ether',
    'fr_ketone', 'fr_ketone_Topliss', 'fr_methoxy', 'fr_para_hydroxylation', 'fr_phenol',
    'fr_phenol_noOrthoHbond', 'fr_unbrch_alkane',
]

_CALC = MolecularDescriptorCalculator(FEATURES)


def featurize(smiles):
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES: {smi!r}")
        rows.append(_CALC.CalcDescriptors(mol))
    return pd.DataFrame(rows, columns=FEATURES)


def evaluate(prop):
    test = pd.read_csv(os.path.join(DATA, f"test_{prop}.csv"))
    model = load(os.path.join(HERE, f"best_xgboost_model_{prop}.joblib"))
    scaler = load(os.path.join(HERE, f"scaler_{prop}.joblib"))

    X = scaler.transform(featurize(test["SMILES"]))
    y_pred = model.predict(X)
    y_true = test[prop]
    return r2_score(y_true, y_pred), mean_squared_error(y_true, y_pred) ** 0.5, len(y_true)


def main():
    print(f"{'Property':28s} {'n_test':>7s} {'R2':>8s} {'RMSE':>10s}")
    print("-" * 56)
    for prop in PROPERTIES:
        r2, rmse, n = evaluate(prop)
        print(f"{prop:28s} {n:7d} {r2:8.3f} {rmse:10.4f}")
    print("\nNote: viscosity is modelled as log10(kinematic viscosity in cSt),")
    print("so its R2/RMSE are reported in log space, as in the paper.")


if __name__ == "__main__":
    main()
