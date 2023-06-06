# stock_scanner_app/views.py code :
from django.shortcuts import render
from .forms import ScannerForm
from scanners.scanner_factory import ScannerFactory
from stock_scanner_app.models import HistoricalData
import pandas as pd
from .models import CompanyInfo
from concurrent.futures import ThreadPoolExecutor
import time
from django.core.cache import cache
from scanners.scanner_preferences import get_preferences
import plotly.io as pio
from charts.chart_factory import ChartFactory
import json
import plotly
import re

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
    historical_data = HistoricalData.objects.select_related('company').filter(symbol=symbol)
    data = pd.DataFrame.from_records(historical_data.values())

    if preferences['timeframe'] == '1D':
        resampled_data = data
    else:
        resampled_data = resample_data(data, preferences['timeframe'])

    if resampled_data.empty:
        return None

    required_columns = ['high', 'low', 'close', 'open']
    missing_columns = [col for col in required_columns if col not in resampled_data.columns]
    if missing_columns:
        return None

    resampled_data['symbol'] = symbol
    scanner = ScannerFactory.create_scanner(scanner_type, preferences)
    result = scanner.scan(resampled_data)
    return result

def get_symbols(selected_exchange, selected_sector):
    if selected_sector and selected_sector != "Whole NSE":
        symbols = CompanyInfo.objects.select_related('sector').filter(sector=selected_sector).values_list('symbol', flat=True)
    else:
        symbols = CompanyInfo.objects.select_related('exchange').filter(exchange=selected_exchange).values_list('symbol', flat=True)
    return symbols


def process_and_cache_results(symbols, preferences, scanner_type):
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(lambda symbol: process_symbol(symbol, preferences, scanner_type), symbols))

    results = [result for result in results if result is not None]
    try:
        results = pd.concat(results)
    except ValueError:
        results = pd.DataFrame()

    return results

def get_chart_data(results_list):
    chart_data = {}
    for result in results_list:
        symbol = result['symbol']

        if symbol not in chart_data:
            historical_data = HistoricalData.objects.filter(symbol=symbol).order_by('-date')[:70][::-1]
            chart_data[symbol] = {'Open': [], 'High': [], 'Low': [], 'Close': [], 'Date': []}

            for data in historical_data:
                chart_data[symbol]['Open'].append(data.open)
                chart_data[symbol]['High'].append(data.high)
                chart_data[symbol]['Low'].append(data.low)
                chart_data[symbol]['Close'].append(data.close)
                chart_data[symbol]['Date'].append(data.date)

    return chart_data

def generate_charts(chart_data):
    for symbol, data in chart_data.items():
        df = pd.DataFrame(data)
        chart_obj = ChartFactory.create_chart('candlestick', df, title=f'{symbol} Candlestick Chart')
        chart = chart_obj.create_chart()
        chart_data[symbol] = {'data': chart.data, 'layout': chart.layout}

    return chart_data

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
            selected_exchange = form.cleaned_data['exchange']
            selected_sector = form.cleaned_data['sector']
            preferences = get_preferences(form, scanner_type)
            symbols = get_symbols(selected_exchange, selected_sector)

            cache_key = f"results_{scanner_type}_{selected_exchange}_{selected_sector}_{'_'.join(map(str, preferences.values()))}"
            
            results = cache.get(cache_key)

            if results is None:
                results = process_and_cache_results(symbols, preferences, scanner_type)
                cache.set(cache_key, results, 300)  # Cache the results for 5 minutes (300 seconds)

            empty_results = results.empty
            results_list = [dict(row._asdict()) for row in results.itertuples()]



            
            # Create a candlestick chart for each symbol
            chart_data = {}
            for result in results_list:
                symbol = result['symbol']
                
                if symbol not in chart_data:
                    historical_data = HistoricalData.objects.filter(symbol=symbol).order_by('-date')[:70][::-1]
                    chart_data[symbol] = {'Open': [], 'High': [], 'Low': [], 'Close': [], 'Date': []}
                    
                    for data in historical_data:
                        chart_data[symbol]['Open'].append(data.open)
                        chart_data[symbol]['High'].append(data.high)
                        chart_data[symbol]['Low'].append(data.low)
                        chart_data[symbol]['Close'].append(data.close)
                        chart_data[symbol]['Date'].append(data.date)

            for symbol, data in chart_data.items():
                df = pd.DataFrame(data)
                chart_obj = ChartFactory.create_chart('candlestick', df, title=f'{symbol} Candlestick Chart')
                chart = chart_obj.create_chart()
                chart_data[symbol] = {'data': chart.data, 'layout': chart.layout}

            context = {'form': form, 'results': results_list, 'columns': results.columns, 'empty_results': empty_results, 'chart_data': json.dumps(chart_data, cls=plotly.utils.PlotlyJSONEncoder)}



        else:
            context = {'form': form}
    else:
        form = ScannerForm()
        context = {'form': form}

    return render(request, 'stock_scanner_app/index.html', context)

