# train.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from feature_engineering import FEATURE_ORDER  # même ordre qu'à l'inférence — source unique


def train(csv_path: str = "training_data.csv"):
    df = pd.read_csv(csv_path)

    X = df[FEATURE_ORDER]   # <- plus de liste dupliquée à la main
    y = df["accepted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)

    print(classification_report(y_test, model.predict(X_test)))

    # on sauvegarde le modèle ET l'ordre des features utilisé, pour pouvoir
    # vérifier au chargement (dans main.py) qu'on ne charge jamais un modèle
    # entraîné avec un ordre de colonnes différent de celui utilisé aujourd'hui.
    joblib.dump({"model": model, "features": FEATURE_ORDER}, "models/ranking_model.pkl")
    print(f"✅ Modèle sauvegardé avec {len(FEATURE_ORDER)} features dans l'ordre : {FEATURE_ORDER}")


if __name__ == "__main__":
    train()