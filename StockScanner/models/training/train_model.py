
from sklearn.model_selection import train_test_split
from models.preprocessing import load_data, preprocess_data
from models.feature_extraction import extract_features
from models.algorithm_selection import select_algorithms
from datetime import datetime
import os
import joblib


def train_models():
    data = preprocess_data()
    X, y = extract_features(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = select_algorithms()
    trained_models = []

    for model in models:
        model.fit(X_train, y_train)
        save_model(model)
        trained_models.append((model, X_test, y_test))

    return trained_models


import os

def save_model(model):
    # Save the trained model to a file
    timestamp = datetime.now().strftime('%Y%m%d_%H_%M')
    model_dir = 'stock_scanner_app/models/trained_models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_file = f'{model_dir}/model_{timestamp}.joblib'
    joblib.dump(model, model_file)

