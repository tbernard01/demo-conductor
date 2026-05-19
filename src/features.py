import numpy as np


def add_features(X, feature_names=None):
    """Applique une feature selection sur X.

    TODO: implémenter une vraie sélection de features.
    Pour l'instant, renvoie X tel quel.
    """
    return X


if __name__ == "__main__":
    from preprocess import load_data, preprocess
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    df_raw = load_data()
    X, y = preprocess(df_raw)
    X_feat = add_features(X, feature_names=data.feature_names)
    print(f"Features shape: {X_feat.shape}")
