import pandas as pd
from stock_scanner_app.models import HistoricalData

def load_data():
    # Load data from the HistoricalData model
    pass

def preprocess_data():
    data = pd.DataFrame(list(HistoricalData.objects.values()))
    print(data)
    data['doji'] = data.apply(lambda row: abs(row['open'] - row['close']) <= 0.05 * (row['high'] - row['low']), axis=1)
    data['body_range'] = abs(data['open'] - data['close'])
    data['high_low_range'] = data['high'] - data['low']
    return data[['body_range', 'high_low_range', 'doji']]