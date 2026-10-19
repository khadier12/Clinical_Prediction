import os
import joblib
import pandas as pd
from Core.model import ClinicalPredictor

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'Data', 'model.pkl')
    scaler_path = os.path.join(base_dir, 'Data', 'scaler.pkl')

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("❌ Erreur : Veuillez d'abord entraîner le modèle avec : python -m pipeline.trainer")
        return

    # Chargement
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    predictor = ClinicalPredictor(model=model, scaler=scaler)

    # Données de test pour deux patients
    nouveaux_patients = pd.DataFrame([
        {'Pregnancies': 1, 'Glucose': 85, 'BloodPressure': 66, 'BMI': 22.0, 'Age': 25},
        {'Pregnancies': 5, 'Glucose': 160, 'BloodPressure': 85, 'BMI': 34.5, 'Age': 50}
    ])

    for idx, row in nouveaux_patients.iterrows():
        patient_df = pd.DataFrame([row])
        
        # 1. Obtenir la probabilité depuis ClinicalPredictor
        prob = predictor.diagnose(patient_df)
        
        # 2. Appliquer la condition dans main.py
        statut = "Infecté" if prob >= 0.5 else "sain"
        
        print(f"Patient {idx + 1} : Probabilité = {prob:.4f} -> Statut : {statut}")

if __name__ == "__main__":
    main()