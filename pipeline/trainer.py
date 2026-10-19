import os
import joblib
from sklearn.linear_model import LogisticRegression


def train_model():
    # Définition des chemins absolus vers le dossier Data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'Data')
    
    # 1. S'assurer que le dossier Data/ existe, sinon le créer
    os.makedirs(data_dir, exist_ok=True)

    data_path = os.path.join(data_dir, 'patient.csv')
    model_path = os.path.join(data_dir, 'model.pkl')
    scaler_path = os.path.join(data_dir, 'scaler.pkl')

    # 2. Préparation et entraînement
    X_train, X_test, y_train, y_test, scaler = prepare_data(data_path)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 3. Sauvegarde dans le dossier Data/
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"✅ Modèle créé et sauvegardé sous : {model_path}")
    print(f"✅ Scaler créé et sauvegardé sous : {scaler_path}")

    return model, scaler, X_test, y_test

if __name__ == "__main__":
    train_model()