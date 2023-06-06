# patterns/utils/data_preprocessing.py

import pandas as pd
from stock_scanner_app.models import HistoricalData
from sklearn.preprocessing import MinMaxScaler

def load_data():
    # Load data from the HistoricalData model
    queryset = HistoricalData.objects.values()
    df = pd.DataFrame.from_records(queryset)
    
    # Convert date column to datetime dtype
    df['date'] = pd.to_datetime(df['date'])
    
    # Set symbol and date columns as index
    df.set_index(['symbol', 'date'], inplace=True)
    
    # Remove rows with null values
    df.dropna(inplace=True)
    
    # Remove rows with no data (i.e., all columns are NaN)
    df.dropna(how='all', inplace=True)
    print(df)
    return df



def normalize_features(df, feature_columns):
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    return df



