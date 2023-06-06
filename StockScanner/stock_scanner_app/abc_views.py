from django.shortcuts import render
from .forms import ScannerForm
from scanners.scanner_factory import ScannerFactory
from stock_scanner_app.models import HistoricalData
import pandas as pd
from .models import CompanyInfo

# Function to resample data
def resample_data(data, timeframe):
    if 'date' not in data.columns:
        return pd.DataFrame()  # Return an empty DataFrame if the 'date' column is missing

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

# Main view function
def index(request):
    if request.method == 'POST':
        form = ScannerForm(request.POST)
        if form.is_valid():
            # Extract form data
            scanner_type = form.cleaned_data['scanner_type']
            selected_exchange = form.cleaned_data['exchange']
            selected_sector = form.cleaned_data['sector']
            preferences = {
                'timeframe': form.cleaned_data['timeframe'],
                'price_limit': form.cleaned_data['price_limit'],
            }

            # Retrieve symbols based on selected sector or exchange
            if selected_sector and selected_sector != "Whole NSE":
                symbols = CompanyInfo.objects.filter(sector=selected_sector).values_list('symbol', flat=True)
            else:
                symbols = CompanyInfo.objects.filter(exchange=selected_exchange).values_list('symbol', flat=True)

            # Set preferences for doji scanner
            if scanner_type == 'doji':
                preferences['tolerance'] = form.cleaned_data['doji_tolerance']

            # Set preferences for Spike Volume scanner
            if scanner_type == 'spike_volume':
                preferences['threshold'] = form.cleaned_data['spike_threshold']

            # Process symbols and run scanner
            results = []
            for symbol in symbols:
                historical_data = HistoricalData.objects.filter(company__symbol=symbol)
                data = pd.DataFrame.from_records(historical_data.values())
                resampled_data = resample_data(data, preferences['timeframe'])

                # Skip the symbol if the resampled_data is empty
                if resampled_data.empty:
                    print(f"Skipping {symbol} because there is no data.")
                    continue

                # Skip the symbol if any required column is missing
                required_columns = ['high', 'low', 'close', 'open']
                missing_columns = [col for col in required_columns if col not in resampled_data.columns]
                if missing_columns:
                    print(f"Skipping {symbol} because the following columns are missing: {', '.join(missing_columns)}")
                    continue

                resampled_data['symbol'] = symbol  # Add the symbol column to the resampled_data DataFrame
                scanner = ScannerFactory.create_scanner(scanner_type, preferences)
                result = scanner.scan(resampled_data)
                results.append(result)

            # Combine results and update context
            results = pd.concat(results)
            
            print(results)

            # Check the type
            print("\nType of the concatenated results:", type(results))

            # Check the shape
            print("\nShape of the concatenated results:", results.shape)

            # Check the data types of the columns
            print("\nData types of the concatenated results:")
            print(results.dtypes)

            context = {'form': form, 'results': results}
        else:
            context = {'form': form}
    else:
        form = ScannerForm()
        context = {'form': form}

    return render(request, 'stock_scanner_app/index.html', context)