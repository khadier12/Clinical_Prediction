import os
import numpy as np
import pandas as pd

def generate_and_save_dataset(file_path=None):
    """
    Génère 200 lignes de données synthétiques sur le diabète
    et les enregistre dans le fichier patient.csv situé dans le dossier Data.
    """
    # Résolution automatique du chemin vers Data/patient.csv si non spécifié
    if file_path is None:
        # Récupère le dossier racine du projet (GL_Clinical)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, 'Data', 'patient.csv')

    np.random.seed(42)
    n_samples = 200

    # 1. Génération des données synthétiques
    age = np.random.randint(18, 80, size=n_samples)
    pregnancies = np.random.choice(range(0, 10), size=n_samples, p=[0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.04, 0.03, 0.02, 0.01])
    bmi = np.round(np.clip(np.random.normal(28, 6, size=n_samples), 18.0, 50.0), 1)
    blood_pressure = np.clip(np.random.normal(72, 12, size=n_samples).astype(int), 50, 120)
    glucose = np.clip(np.random.normal(110, 30, size=n_samples).astype(int), 60, 200)

    # Calcul du risque/cible
    logit = -8 + (0.04 * glucose) + (0.08 * bmi) + (0.03 * age) + (0.2 * pregnancies)
    probability = 1 / (1 + np.exp(-logit))
    outcome = (np.random.uniform(0, 1, size=n_samples) < probability).astype(int)

    # 2. Création du DataFrame
    df = pd.DataFrame({
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'BMI': bmi,
        'Age': age,
        'Outcome': outcome
    })

    # 3. Écriture dans le fichier Data/patient.csv
    df.to_csv(file_path, index=False)
    print(f"✅ Données enregistrées avec succès dans : {file_path}")

    return df

if __name__ == "__main__":
    generate_and_save_dataset()
    