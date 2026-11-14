
import os
import joblib
from sklearn.linear_model import LogisticRegression
from utils.preprocessing import prepare_data


def train_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'Data')

    os.makedirs(data_dir, exist_ok=True)

    data_path = os.path.join(data_dir, 'patient.csv')
    model_path = os.path.join(data_dir, 'model.pkl')
    scaler_path = os.path.join(data_dir, 'scaler.pkl')

    # Préparation des données
    X_train, X_test, y_train, y_test, scaler = prepare_data(data_path)

    # Entraînement
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Sauvegarde
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Modèle sauvegardé : {model_path}")
    print(f"Scaler sauvegardé : {scaler_path}")

    return model, scaler, X_test, y_test


if __name__ == "__main__":
    train_model()
