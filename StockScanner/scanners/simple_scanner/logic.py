import pandas as pd

class SimpleScanner:
    def __init__(self, exchange=None, sector=None, timeframe=None, **kwargs):
        self.exchange = exchange
        self.sector = sector
        self.timeframe = timeframe

    def scan(self, historical_data, preferences=None):
        result = historical_data.drop(['company_id', 'id'], axis=1)
        result = result.head(10)

       

        return result
