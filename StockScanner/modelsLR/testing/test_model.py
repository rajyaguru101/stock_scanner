from sklearn.metrics import accuracy_score, classification_report
from training.train_model import train_model
from datetime import datetime
import os

def test_model():
    model, X_test, y_test = train_model()
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Add a return statement here
    return accuracy, report

    # Save the results to a file
    save_results(results)


def save_results(results):
    # Save the results to a CSV file
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
    result_dir = 'results'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_file = f'{result_dir}/model_1_{timestamp}.csv'

    with open(result_file, 'w') as f:
        f.write(f'Accuracy: {results["accuracy"]}\n')
        f.write(f'{results["report"]}\n')

