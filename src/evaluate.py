from sklearn.metrics import accuracy_score


def evaluate(model, X_test, y_test):
    """Évalue le modèle sur le jeu de test.

    TODO: ajouter precision, recall, F1 et matrice de confusion.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    return {"accuracy": acc}


if __name__ == "__main__":
    from preprocess import load_data, preprocess
    from features import add_features
    from train import train

    df = load_data()
    X, y = preprocess(df)
    X_feat = add_features(X)
    model, X_test, y_test = train(X_feat, y)
    results = evaluate(model, X_test, y_test)
    print(results)
