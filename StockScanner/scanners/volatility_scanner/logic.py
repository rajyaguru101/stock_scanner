import pandas as pd

class VolatilityScanner:
    def __init__(self, preferences):
        self.atr_multiplier = float(preferences['atr_multiplier'])  # Convert to float
        self.atr_window_size = preferences['atr_window_size']

    @staticmethod
    def average_true_range(high, low, close, window):
        hl = high - low
        hc = (high - close.shift(1)).abs()
        lc = (low - close.shift(1)).abs()

        true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()

        return atr

    def scan(self, data):
        
        data['atr'] = self.average_true_range(data['high'], data['low'], data['close'], self.atr_window_size)
        data['atr_percentage'] = data['atr'] / data['close']

        high_vol_stocks = data[data['atr_percentage'] > self.atr_multiplier]
        last_rows = high_vol_stocks.groupby('symbol').last().reset_index()

        return last_rows
