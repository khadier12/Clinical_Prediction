import pandas as pd
import numpy as np

class ClinicalPredictor:
    def __init__(self, model, scaler=None):
        """
        Initialise le prédicteur clinique avec un modèle pré-entraîné et un scaler optionnel.
        :param model: Modèle de Machine Learning entraîné (ex: LogisticRegression)
        :param scaler: Scaler optionnel pour normaliser les données avant prédiction
        """
        self.model = model
        self.scaler = scaler

    def diagnose(self, patient_data):
        """
        Calcule et retourne la probabilité d'infection pour un patient.
        :param patient_data: Données du patient (DataFrame pandas, numpy array 2D ou dictionnaire)
        :return: float (probabilité entre 0 et 1)
        """
        # Convertir en DataFrame si l'entrée est un dictionnaire
        if isinstance(patient_data, dict):
            patient_data = pd.DataFrame([patient_data])

        # Appliquer la normalisation si un scaler est fourni
        if self.scaler is not None:
            patient_scaled = self.scaler.transform(patient_data)
        else:
            patient_scaled = patient_data

        # Obtenir la probabilité de la classe positive
        if hasattr(self.model, "predict_proba"):
            probability = self.model.predict_proba(patient_scaled)[0][1]
        else:
            probability = self.model.predict(patient_scaled)[0]

        return probability