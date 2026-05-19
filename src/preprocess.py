import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


def load_data():
    """Charge le dataset breast cancer sous forme de DataFrame."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    # BUG: aucune gestion des valeurs nulles
    # Si le dataset contient des nulls, les étapes suivantes planteront.
    return df


def preprocess(df):
    """Normalise les features."""
    X = df.drop("target", axis=1)
    y = df["target"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y.values


if __name__ == "__main__":
    df = load_data()
    X, y = preprocess(df)
    print(f"Data shape: {X.shape}")
