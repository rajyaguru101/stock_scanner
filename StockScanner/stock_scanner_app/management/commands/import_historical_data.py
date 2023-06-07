from datetime import datetime, timedelta
import yfinance as yf
from django.core.management.base import BaseCommand
from stock_scanner_app.models import CompanyInfo, HistoricalData

class Command(BaseCommand):
    help = 'Imports historical data for NSE symbols using yfinance'

    def add_arguments(self, parser):
        parser.add_argument('--end_date', type=str, default=None, help='End date for historical data (YYYY-MM-DD)')

    def handle(self, *args, **options):
        end_date = options['end_date']

        symbols = CompanyInfo.objects.values_list('symbol', flat=True)

        for symbol in symbols:
            # Get the latest date available in the HistoricalData table for this symbol
            latest_date = HistoricalData.objects.filter(company__symbol=symbol).order_by('-date').values('date').first()

            if latest_date:
                start_date = latest_date['date'] + timedelta(days=1)
            else:
                start_date = '2021-02-24'

            data = yf.download(symbol, start=start_date, end=end_date)

            # Check if yfinance provided data
            if not data.empty:
                for date, row in data.iterrows():
                    historical_data = HistoricalData(
                        company=CompanyInfo.objects.get(symbol=symbol),
                        date=date,
                        open=row['Open'],
                        high=row['High'],
                        low=row['Low'],
                        close=row['Close'],
                        volume=row['Volume']
                    )
                    historical_data.symbol = symbol  # Set the symbol field here
                    historical_data.save()

        self.stdout.write(self.style.SUCCESS('Successfully imported historical data'))

# COMMANDS
# python manage.py import_historical_data
# python manage.py import_historical_data --end_date=2023-06-06
