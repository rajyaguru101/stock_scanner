from .simple_scanner.logic import SimpleScanner
from .spike_volume.logic import SpikeVolumeScanner
from .moving_average_crossover.logic import MovingAverageCrossoverScanner
from .doji.logic import DojiScanner
from .volatility_scanner.logic import VolatilityScanner
from .percentage_stock_scanner.logic import PercentageStockScanner

class ScannerFactory:
    @staticmethod
    def create_scanner(scanner_type, preferences=None):
        if scanner_type == 'simple_scanner':
            return SimpleScanner(preferences=preferences)
        elif scanner_type == 'spike_volume':
            return SpikeVolumeScanner(preferences=preferences)
        elif scanner_type == 'moving_average_crossover':
            return MovingAverageCrossoverScanner(preferences=preferences)
        elif scanner_type == 'doji':
            return DojiScanner(preferences=preferences)
        elif scanner_type == 'volatility':
            return VolatilityScanner(preferences)
        elif scanner_type == 'percentage_stock_scanner':
            return PercentageStockScanner(preferences)
        else:
            raise ValueError(f"Invalid scanner type: {scanner_type}")
