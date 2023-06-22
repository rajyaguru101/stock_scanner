
from .indicators.doji import DojiScanner
from .indicators.engulfing_pattern import EngulfingPattern


class IndicatorFactory:
    @staticmethod
    def create_scanner(candle_type, preferences=None):
        if candle_type == 'SMA':
            return DojiScanner(preferences=preferences)
        elif candle_type == 'moving_average_crossover':
            return EngulfingPattern(preferences)

        else:
            raise ValueError(f"Invalid scanner type: {candle_type}")
