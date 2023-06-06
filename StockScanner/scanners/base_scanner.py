# scanners/base_scanner.py

class BaseScanner:
    def scan(self, data):
        raise NotImplementedError()
