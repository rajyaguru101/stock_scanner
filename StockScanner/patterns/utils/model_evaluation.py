import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from patterns.models.doji import create_lstm_model, train_model, test_model
from patterns.features.doji_features import load_data, preprocess_data
import pandas as pd
from tensorflow.keras.models import load_model
import os
def save_model(model, model_name="doji_lstm_model"):
    model_path = os.path.join("patterns", "trained_model", f"{model_name}.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")

def save_performance_report(test_loss, test_accuracy, model_name="doji_lstm_model"):
    report = pd.DataFrame(
        data={"model": [model_name], "test_loss": [test_loss], "test_accuracy": [test_accuracy]}
    )
    report_path = os.path.join("patterns", "trained_model", f"{model_name}_performance_report.csv")
    report.to_csv(report_path, index=False)
    print(f"Performance report saved to {report_path}")


def evaluate_model():
    # Load and preprocess the data
    df = load_data()
    print(f"Data after loading: {df.shape}")  # Add this line
    df = preprocess_data(df)
    print(f"Data after preprocessing: {df.shape}")  # Add this line

    # Set the target variable (trend reversal)
    df['trend_reversal'] = df['doji'].shift(-1)
    df.dropna(inplace=True)

    # Convert the target variable to integers
    df['trend_reversal'] = df['trend_reversal'].astype(int)

    # Select features and target variable
    feature_columns = ['doji', 'ohlc_ratio', 'volume_pct_change', 'sma', 'ema', 'rsi',
       'ema_fast', 'ema_slow', 'macd', 'macd_signal', 'macd_histogram', 'std',
       'bollinger_upper', 'bollinger_lower', 'bollinger_band_width', 'roc',
       'obv', 'williams_r', '%K', '%D', 'adx', 'aroon_up', 'aroon_down', 'cci',
       'parabolic_sar', 'keltner_upper', 'keltner_lower']  # Add the feature column names here
    target_column = 'trend_reversal'

    X = df[feature_columns].values
    y = df[target_column].values

    # Scale the features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape the input for the LSTM model
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    # Create and train the LSTM model
    input_shape = (X_train.shape[1], X_train.shape[2])
    lstm_model = create_lstm_model(input_shape)
    lstm_model = train_model(lstm_model, X_train, y_train)

    # Evaluate the model on the test set
    test_loss, test_accuracy = test_model(lstm_model, X_test, y_test)

    # Save the trained model
    save_model(lstm_model)

    # Save the performance report
    save_performance_report(test_loss, test_accuracy)

if __name__ == "__main__":
    evaluate_model()
