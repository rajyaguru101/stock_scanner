#scanners/doji/logic.py
from scanners.base_scanner import BaseScanner
from stock_scanner_app.models import HistoricalData
from decimal import Decimal
import pandas as pd

class DojiScanner:
    def __init__(self, preferences):
        self.tolerance = Decimal(preferences['doji_tolerance'])

    def scan(self, historical_data):

        historical_data['range'] = historical_data['high'] - historical_data['low']
        historical_data['mean'] = historical_data.groupby('symbol')['range'].transform('mean')
        historical_data = historical_data.sort_values('date', ascending=False)
        historical_data_last = historical_data.groupby('symbol').tail(1)
        
        historical_data_last['body'] = abs(historical_data_last['close'] - historical_data_last['open'])

        doji_tolerance = historical_data_last['body'] / historical_data_last['range']
        
        # Convert doji_tolerance Series to Decimal values
        doji_tolerance = doji_tolerance.apply(Decimal)

        if (doji_tolerance <= self.tolerance).all():
            historical_data_last['doji'] = 1
            print(historical_data_last)
            results = historical_data_last.dropna()

            return results
        else:
            return pd.DataFrame()



    def get_data(self, symbols):
        # Fetch historical data for all symbols at once
        historical_data = HistoricalData.objects.select_related('company').only('symbol', 'date', 'open', 'high', 'low', 'close').filter(symbol__in=symbols)
        data = pd.DataFrame.from_records(historical_data.values())

        if 'date' not in data.columns:
            return pd.DataFrame()

        data['date'] = pd.to_datetime(data['date'])  # convert date column to datetime format
        return data
