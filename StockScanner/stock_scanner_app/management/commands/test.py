from datetime import datetime
import yfinance as yf
from django.core.management.base import BaseCommand
from stock_scanner_app.models import CompanyInfo, HistoricalData

class Command(BaseCommand):
    help = 'Imports historical data for NSE symbols using yfinance'

    def add_arguments(self, parser):
        parser.add_argument('start_date', type=str, help='Start date for historical data (YYYY-MM-DD)')
        parser.add_argument('end_date', type=str, help='End date for historical data (YYYY-MM-DD)')

    def handle(self, *args, **options):
        start_date = options['start_date']
        end_date = options['end_date']

        symbols = CompanyInfo.objects.values_list('symbol', flat=True)

        for symbol in symbols:
            data = yf.download(symbol, start=start_date, end=end_date)
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
                historical_data.save()

        self.stdout.write(self.style.SUCCESS('Successfully imported historical data'))


# python manage.py import_historical_data 2023-01-01 2023-03-24