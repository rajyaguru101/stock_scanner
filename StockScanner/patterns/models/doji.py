import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from imblearn.over_sampling import SMOTE
import joblib
from patterns.features.doji_features import load_data, generate_features, identify_dojis, label_trend_reversals
from sklearn.impute import SimpleImputer


def prepare_data():
    df = load_data()
    df = generate_features(df)
    df['doji'] = identify_dojis(df)
    df = label_trend_reversals(df)

    features = df[['SMA_5', 'SMA_10', 'RSI', 'volatility', 'bollinger_upper', 'bollinger_lower', 'macd', 'macd_signal', 'upper_shadow', 'lower_shadow', 'body_length', 'shadow_body_ratio', 'roc', 'adx', 'aroon_up', 'aroon_down', 'cmf', 'stoch_k', 'stoch_d']].values
    labels = df['reversal'].values

    return features, labels

def preprocess_data(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

    # Handle missing values using SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    return X_train_resampled, X_test, y_train_resampled, y_test, scaler

def create_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(19, 1)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

def evaluate_model(model, X_test, y_test):
   
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("Confusion Matrix:\n", cm)


def make_prediction(symbol, features):
    df = load_data()
    df = generate_features(df)
    df = df[df['symbol'] == symbol].tail(1)
    features = df[['SMA_5', 'SMA_10', 'RSI', 'volatility', 'bollinger_upper', 'bollinger_lower', 'macd', 'macd_signal', 'upper_shadow', 'lower_shadow', 'body_length', 'shadow_body_ratio', 'roc']].values

    # Load the saved model
    model = tf.keras.models.load_model('doji_model.h5')
    scaler = joblib.load('doji_scaler.pkl')

    # Preprocess input features
    normalized_features = scaler.transform([features])
    normalized_features = normalized_features.reshape(normalized_features.shape[0], normalized_features.shape[1], 1)

    # Make the prediction
    prediction = model.predict_classes(normalized_features)
    return prediction

if __name__ == "__main__":
    features, labels = prepare_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(features, labels)

    model = create_model()
    train_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Save the model and the scaler
    model.save('doji_model.h5')
    joblib.dump(scaler, 'doji_scaler.pkl')

    # Example usage: making a prediction for a specific stock
    symbol = "20MICRONS.NS"
    prediction = make_prediction(symbol, features)
    print(f"Predicted trend reversal for {symbol}: {prediction[0][0]}")
