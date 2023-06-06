from .simple_scanner.logic import SimpleScanner
from .spike_volume.logic import SpikeVolumeScanner
from .moving_average_crossover.logic import MovingAverageCrossoverScanner
from .doji.logic import DojiScanner



def get_scanner(scanner_type):
    scanners = {
        'simple_scanner': SimpleScanner,
        'spike_volume': SpikeVolumeScanner,
        'moving_average_crossover': MovingAverageCrossoverScanner,
        'doji': DojiScanner,
    }

    if scanner_type not in scanners:
        raise ValueError(f"Invalid scanner_type: {scanner_type}")

    return scanners[scanner_type]
