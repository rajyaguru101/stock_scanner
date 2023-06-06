from sklearn.metrics import accuracy_score, classification_report
from models.training.train_model import train_models



from datetime import datetime
import os

def test_model():
    trained_models = train_models()

    results = []

    for model, X_test, y_test in trained_models:
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        results.append({"model": type(model).__name__, "accuracy": accuracy, "report": report})

    # Save the results
    for result in results:
        save_results(result)
        # Return the accuracy and report for the first model in the list
    return results[0]["accuracy"], results[0]["report"]


def save_results(results):
    timestamp = datetime.now().strftime('%Y%m%d_%H_%M')
    result_dir = 'stock_scanner_app/results'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_file = f'{result_dir}/{results["model"]}_{timestamp}.csv'

    with open(result_file, 'w') as f:
        f.write(f'Accuracy: {results["accuracy"]}\n')
        f.write(f'{results["report"]}\n')



