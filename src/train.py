from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def train(X, y):
    """Entraîne un XGBoost et renvoie le modèle + split de test."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = XGBClassifier(
        n_estimators=100,
        random_state=42,
        objective="binary:logistic",
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model, X_test, y_test


if __name__ == "__main__":
    from preprocess import load_data, preprocess
    from features import add_features

    df = load_data()
    X, y = preprocess(df)
    X_feat = add_features(X)
    model, X_test, y_test = train(X_feat, y)
    print("Modèle entraîné !")
