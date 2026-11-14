
import os
import joblib
import numpy as np


class ClinicalPredictor:

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "Data")

        model_path = os.path.join(data_dir, "model.pkl")
        scaler_path = os.path.join(data_dir, "scaler.pkl")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, patient_data):

        # Respecter l'ordre des colonnes du CSV
        features = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "BMI",
            "Age"
        ]

        X = np.array([[
            patient_data[col] for col in features
        ]])

        # Appliquer la même normalisation que pendant l'entraînement
        X_scaled = self.scaler.transform(X)

        # Prédiction
        prediction = self.model.predict(X_scaled)[0]

        # Probabilité si disponible
        probability = None

        if hasattr(self.model, "predict_proba"):
            probability = self.model.predict_proba(X_scaled)[0][1]

        return {
            "prediction": int(prediction),
            "result": "Diabétique" if prediction == 1 else "Non diabétique",
            "probability": float(probability) if probability is not None else None
        }


if __name__ == "__main__":

    predictor = ClinicalPredictor()

    patient = {
        "Pregnancies": 6,
        "Glucose": 134,
        "BloodPressure": 91,
        "BMI": 18.0,
        "Age": 56
    }

    result = predictor.predict(patient)

    print("\nRésultat de la prédiction :")
    print(result)
