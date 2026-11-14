
import os
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from utils.preprocessing import prepare_data


def evaluate_model():

    # 1. Définir les chemins
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "Data")

    data_path = os.path.join(data_dir, "patient.csv")
    model_path = os.path.join(data_dir, "model.pkl")

    # 2. Vérifier que le modèle existe
    if not os.path.exists(model_path):
        print("Erreur : modèle introuvable. Lance d'abord trainer.py")
        return

    # 3. Préparer les données de test
    X_train, X_test, y_train, y_test, scaler = prepare_data(data_path)

    # 4. Charger le modèle entraîné
    model = joblib.load(model_path)

    # 5. Effectuer les prédictions
    y_pred = model.predict(X_test)

    # 6. Calculer les métriques
    accuracy = accuracy_score(y_test, y_pred)

    print("\n========== EVALUATION DU MODELE ==========")

    print(f"\nAccuracy : {accuracy:.4f}")

    print("\nRapport de classification :")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Non diabétique", "Diabétique"],
        zero_division=0
    ))

    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred))

    # 7. AUC-ROC
    if len(set(y_test)) == 2:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"\nAUC-ROC : {auc:.4f}")


if __name__ == "__main__":
    evaluate_model()
