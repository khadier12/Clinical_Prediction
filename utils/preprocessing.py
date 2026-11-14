
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data(data_path):
    # 1. Charger le dataset
    df = pd.read_csv(data_path)

    # 2. Séparer les variables explicatives et la cible
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # 3. Division : 80% entraînement et 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 4. Normalisation des variables
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler
