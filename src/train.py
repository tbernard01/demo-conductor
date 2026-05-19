from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train(X, y):
    """Entraîne un RandomForest et renvoie le modèle + split de test."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

    # TODO: essayer XGBoost et comparer les deux modèles


if __name__ == "__main__":
    from preprocess import load_data, preprocess
    from features import add_features

    df = load_data()
    X, y = preprocess(df)
    X_feat = add_features(X)
    model, X_test, y_test = train(X_feat, y)
    print("Modèle entraîné !")
