
# scanners/percentage_stock_scanner/logic.py
from scanners.base_scanner import BaseScanner
from stock_scanner_app.models import HistoricalData
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import time

class PercentageStockScanner(BaseScanner):
    def __init__(self, preferences):
        self.percentage_stock_scanner = preferences['percentage_stock_scanner']

    def scan(self, historical_data):
        # Group by 'symbol' and keep only the last two rows for each unique symbol
        last_two_dates_data = historical_data.groupby('symbol').tail(2)

        # Reset the index
        last_two_dates_data.reset_index(drop=True, inplace=True)
        last_two_dates_data = last_two_dates_data.copy()

        # Calculate the percentage change for the 'close' column
        last_two_dates_data['change_pct'] = last_two_dates_data.groupby('symbol')['close'].pct_change()

        # Convert the percentage change to a decimal percentage value
        last_two_dates_data['change_pct'] = last_two_dates_data['change_pct'] * 100

        # Convert 'change_pct' column to float
        last_two_dates_data['change_pct'] = last_two_dates_data['change_pct'].astype(float)

        # Drop NaN rows
        last_two_dates_data.dropna(inplace=True)

        # Filter the DataFrame based on the absolute value of the given percentage
        result = last_two_dates_data[np.abs(last_two_dates_data['change_pct']) >= self.percentage_stock_scanner]
        
        # to remove the ".NS" suffix from the symbol column
        result['symbol'] = result['symbol'].str.replace('.NS', '')
        return result



    def fetch_historical_data(self, symbols_chunk):
        historical_data = HistoricalData.objects.select_related('company').only('symbol', 'date', 'close').filter(symbol__in=symbols_chunk)
        return pd.DataFrame.from_records(historical_data.values())

    def process_symbol_group(self, symbol_group):
        symbol, group = symbol_group
        if group['date'].size >= 2:
            return group.tail(2)  # append the last two rows for each valid symbol
        return None

    def get_data(self, symbols):
        start_time = time.time()  # record the start time

        chunk_size = 10  # You can adjust the chunk size based on your performance needs

        # Fetch historical data for chunks of symbols in parallel
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.fetch_historical_data, symbols[i:i + chunk_size]) for i in range(0, len(symbols), chunk_size)]
            data_chunks = [future.result() for future in as_completed(futures)]

        data = pd.concat(data_chunks, ignore_index=True)

        if 'date' not in data.columns:
            return pd.DataFrame()

        data['date'] = pd.to_datetime(data['date'])  # convert date column to datetime format
        symbol_groups = data.groupby('symbol')

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_symbol_group, symbol_group) for symbol_group in symbol_groups]

            # Collect the DataFrames in a list
            results = [future.result() for future in as_completed(futures)]

        # Concatenate the DataFrames
        valid_data = pd.concat(results, ignore_index=True)
        

        valid_data = valid_data.drop(columns=['id', 'company_id'])  # drop 'id' and 'company_id' columns
        end_time = time.time()  # record the end time
        execution_time = end_time - start_time  # calculate the execution time

        print(f"Execution time for {len(symbols)} symbols: {execution_time:.2f} seconds")  # print the execution time
        return valid_data