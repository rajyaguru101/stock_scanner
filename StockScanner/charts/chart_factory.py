# chart_factory.py
from .candlestick_chart import CandlestickChart

class ChartFactory:
    @staticmethod
    def create_chart(chart_type, *args, **kwargs):
        if chart_type == 'candlestick':
            return CandlestickChart(*args, **kwargs)
        else:
            raise ValueError(f"Invalid chart type: {chart_type}")
