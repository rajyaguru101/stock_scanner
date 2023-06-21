# stock_scanner_app/forms.py code: 
from django import forms
from .models import CompanyInfo

class ScannerForm(forms.Form):
    # Scanner choices
    SCANNER_CHOICES = [
        ('candlesticks', 'Candlesticks'),
        ('simple_scanner', 'Simple Scanner'),
        ('candlestick', 'Candlestick'),
        ('spike_volume', 'Spike Volume'),
        ('moving_average_crossover', 'Moving Acerage Crossover'), 
        ('volatility', 'Volatility'),
        ('percentage_stock_scanner', 'Percentage Stock Scanner'),
    ]

    CANDLESTICK_CHOICES = [
        ('doji', 'Doji'),
        ('engulfing_pattern', 'Engulfing Pattern'),
    ]

    # Exchange choices
    EXCHANGE_CHOICES = [(exchange, exchange) for exchange in CompanyInfo.objects.values_list('exchange', flat=True).distinct()]

    # Sector choices
    SECTOR_CHOICES = [('Whole NSE', 'Whole NSE')] + [(sector, sector) for sector in CompanyInfo.objects.values_list('sector', flat=True).distinct()]

    # Timeframe choices
    TIMEFRAME_CHOICES = [
        ('1D', '1 day'),
        ('2D', '2 days'),
        ('1W', '1 week'),
        ('1M', '1 month'),
    ]

    # Form fields
    scanner_type = forms.ChoiceField(choices=SCANNER_CHOICES, label="Select scanner", widget=forms.Select(attrs={'id': 'scanner_type'}))
    candle_type = forms.ChoiceField(choices=CANDLESTICK_CHOICES, label="Select Candle", widget=forms.Select(attrs={'id': 'candle_type'}))
    exchange = forms.ChoiceField(choices=EXCHANGE_CHOICES, label="Select exchange")
    sector = forms.ChoiceField(choices=SECTOR_CHOICES, required=False, label="Select sector")
    timeframe = forms.ChoiceField(choices=TIMEFRAME_CHOICES, label="Select timeframe")
    spike_threshold = forms.DecimalField(max_digits=10, initial=1.5, required=False, label="Spike Threshold", widget=forms.NumberInput(attrs={'id': 'spike_threshold'}))
    atr_multiplier = forms.DecimalField(label='ATR Multiplier', initial=0.02, required=False, widget=forms.NumberInput(attrs={'id': 'atr_multiplier'}))
    atr_window_size = forms.IntegerField(label='ATR Window Size', initial=5, required=False, widget=forms.NumberInput(attrs={'id': 'atr_window_size'}))
    doji_tolerance = forms.DecimalField(max_digits=10, decimal_places=2, initial=0.3, required=False, label="Tolerance for Doji", widget=forms.NumberInput(attrs={'id': 'doji_tolerance'}))
    percentage_stock_scanner = forms.DecimalField(label='Percentage Stock Scanner', initial=2.00, required=False, widget=forms.NumberInput(attrs={'id': 'percentage_stock_scanner'}))
