#spike_volume/logic.py
import pandas as pd

class SpikeVolumeScanner:
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
        stocks_with_volume_spike = pd.DataFrame(stocks_with_volume_spike)

        return stocks_with_volume_spike



