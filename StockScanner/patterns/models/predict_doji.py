import joblib
import pandas as pd
from patterns.features.doji_features import create_doji_features
from patterns.utils.data_preprocessing import load_data, download_new_data

def predict_doji(model, X):
    return model.predict(X)

def main():
    # Load the trained model
    model = joblib.load("patterns/models/DojiModel.joblib")

    # Download new data
    ticker = "^NSEI"
    start_date = "2021-01-01"
    end_date = "2023-03-30"
    new_data = download_new_data(ticker, start_date, end_date)

    # Create features for new data
    X_new = create_doji_features(new_data)

    # Make predictions on new data
    predictions = predict_doji(model, X_new)

    # Add predictions to the new_data DataFrame and display the results
    new_data["Predicted Doji"] = predictions
    print(new_data.head(6))

if __name__ == "__main__":
    main()