# stock_scanner_app/views.py code :
from django.shortcuts import render
from .forms import ScannerForm
from scanners.scanner_factory import ScannerFactory
from scanners.candlestick import CandlestickFactory
import pandas as pd
from .models import CompanyInfo
from concurrent.futures import ThreadPoolExecutor
import time
from django.core.cache import cache
from scanners.scanner_preferences import get_preferences
import plotly.io as pio
from charts.chart_factory import ChartFactory
import re
from concurrent.futures import ProcessPoolExecutor



def sanitize_cache_key(key):
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", key)


# Function to resample data
def resample_data(data, timeframe):
    if 'date' not in data.columns:
        return pd.DataFrame()  

    data['date'] = pd.to_datetime(data['date'])  # Convert date column to datetime objects
    data.set_index('date', inplace=True)
    resampled_data = data.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    resampled_data.reset_index(inplace=True)
    return resampled_data

def process_symbol(symbol, preferences, scanner_type):
    scanner = ScannerFactory.create_scanner(scanner_type, preferences)
    data = scanner.get_data(symbol)

    if data.empty:
        return None

    required_columns = ['high', 'low', 'close', 'open']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        return None

    data['symbol'] = symbol
    result = scanner.scan(data)
    return result


def get_symbols(selected_exchange, selected_sector):
    if selected_sector and selected_sector != "Whole NSE":
        symbols = CompanyInfo.objects.prefetch_related('sector').filter(sector=selected_sector).values_list('symbol', flat=True)
    else:
        symbols = CompanyInfo.objects.prefetch_related('exchange').filter(exchange=selected_exchange).values_list('symbol', flat=True)
    return symbols


def prepare_context(form, results, chart_data):
    empty_results = results.empty
    results_list = [dict(row._asdict()) for row in results.itertuples()]
    columns = list(results.columns)
    return {'form': form, 'results': results_list, 'columns': columns, 'empty_results': empty_results, 'chart_data': json.dumps(chart_data, cls=plotly.utils.PlotlyJSONEncoder)}


# Main view function
def index(request):
    if request.method == 'POST':
        form = ScannerForm(request.POST)
        if form.is_valid():
            scanner_type = form.cleaned_data['scanner_type']
            candle_type = form.cleaned_data['candle_type']
            selected_exchange = form.cleaned_data['exchange']
            selected_sector = form.cleaned_data['sector']
            
            symbols = get_symbols(selected_exchange, selected_sector)

            results = None

            if results is None:
                if "candlesticks" in scanner_type:
                    preferences = get_preferences(form, candle_type)
                    scanner = CandlestickFactory.create_scanner(candle_type, preferences)
                    
                else:
                    preferences = get_preferences(form, scanner_type)
                    scanner = ScannerFactory.create_scanner(scanner_type, preferences)
                    

                data = scanner.get_data(symbols)  # Fetch data for all symbols
                results = scanner.scan(data)  # Scan the data
                

            empty_results = results.empty
            results_list = [dict(row._asdict()) for row in results.itertuples()]

            context = {'form': form, 'results': results_list, 'columns': results.columns, 'empty_results': empty_results}
        else:
            context = {'form': form}
    else:
        form = ScannerForm()
        context = {'form': form}

    return render(request, 'stock_scanner_app/index.html', context)

