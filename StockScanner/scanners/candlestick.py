
from .candlesticks.doji import DojiScanner
from .candlesticks.engulfing_pattern import EngulfingPattern


class CandlestickFactory:
    @staticmethod
    def create_scanner(candle_type, preferences=None):
        if candle_type == 'doji':
            return DojiScanner(preferences=preferences)
        elif candle_type == 'engulfing_pattern':
            return EngulfingPattern(preferences)

        else:
            raise ValueError(f"Invalid scanner type: {candle_type}")
