import pandas as pd

class MovingAverageCrossoverScanner:

    def __init__(self, exchange=None, sector=None, timeframe=None, **kwargs):
        self.exchange = exchange
        self.sector = sector
        self.timeframe = timeframe

    def scan(self, historical_data, preferences=None):
        data = pd.DataFrame.from_records(historical_data.values())
        self.short_window = 4
        self.long_window = 11
        short_mavg = data.close.rolling(window=self.short_window).mean()
        long_mavg = data.close.rolling(window=self.long_window).mean()
        crossover = (short_mavg > long_mavg) & (short_mavg.shift(1) <= long_mavg.shift(1))
        result = []
        if any(crossover):
            symbol = historical_data[0].company.symbol
            result.append((symbol, crossover))
        return result
