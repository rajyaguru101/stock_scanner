# doji_model.py
import numpy as np

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split



from .models import HistoricalData


def is_doji(row):
    open_close_diff = abs(row['open'] - row['close'])
    high_low_diff = row['high'] - row['low']
    return open_close_diff <= 0.003 * high_low_diff

def prepare_data():
    historical_data = HistoricalData.objects.values()
    data = pd.DataFrame.from_records(historical_data)
    data['doji'] = data.apply(is_doji, axis=1)
    return data


def train_and_test_model():
    data = prepare_data()
    features = data[['open', 'high', 'low', 'close']]
    target = data['doji']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    # Standardize the input features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create a simple neural network
    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=4))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    # Evaluate the model on the test set
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Predict classes and compute confusion matrix
    y_pred = np.round(model.predict(X_test)).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    return model, accuracy, cm


if __name__ == '__main__':
    trained_model, accuracy, cm = train_and_test_model()
    print("Model trained.")
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", cm)
