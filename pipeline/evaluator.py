import os
import joblib
from utils.preprocessing import prepare_data
from utils.metrics import calculate_metrics

def evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'Data', 'patient.csv')
    
    _, X_test, _, y_test, _ = prepare_data(data_path)

    model = joblib.load(os.path.join(base_dir, 'Data', 'model.pkl'))
    y_pred = model.predict(X_test)

    results = calculate_metrics(y_test, y_pred)
    print("📊 Résultats de l'évaluation :", results)
    return results

if __name__ == "__main__":
    evaluate()