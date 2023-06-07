#spike_volume/logic.py
from scanners.base_scanner import BaseScanner
from stock_scanner_app.models import HistoricalData
import pandas as pd

class SpikeVolumeScanner(BaseScanner):
    def __init__(self, preferences):
        print(preferences)
        self.spike_threshold = float(preferences['spike_threshold'])  
        
    def scan(self, historical_data):
        df = historical_data
        volume_spike_threshold = self.spike_threshold 

        # Calculate average volume for each stock symbol
        symbol_avg_volume = df.groupby('symbol')['volume'].mean()
        symbol_avg_volume = round(symbol_avg_volume)

        # Get the last row of each stock symbol
        symbol_last_row = df.groupby('symbol').last().reset_index()

        # Join the average volume data with the last row data
        symbol_last_row = symbol_last_row.join(symbol_avg_volume, on='symbol', rsuffix='_Avg')

        # Check which stocks had a volume spike greater than the threshold
        symbol_last_row["volume_spike"] = symbol_last_row["volume"] > (volume_spike_threshold * symbol_last_row["volume_Avg"])

        # Get the stocks that had a volume spike and sort them by descending volume
        stocks_with_volume_spike = symbol_last_row[symbol_last_row['volume_spike'] == True].sort_values('volume', ascending=False)

        # Set 'Date' as the index again
        result = pd.DataFrame(stocks_with_volume_spike)
        
        # to remove the ".NS" suffix from the symbol column
        result['symbol'] = result['symbol'].str.replace('.NS', '')
        result = result.drop(['id', 'company_id'], axis=1)

        return result
    
    def get_data(self, symbols):
        # Fetch historical data for all symbols at once
        historical_data = HistoricalData.objects.select_related('company').only('symbol', 'date', 'open', 'high', 'low', 'close').filter(symbol__in=symbols)
        data = pd.DataFrame.from_records(historical_data.values())

        if 'date' not in data.columns:
            return pd.DataFrame()

        data['date'] = pd.to_datetime(data['date'])  # convert date column to datetime format
        print(data.head(50))
        return data



