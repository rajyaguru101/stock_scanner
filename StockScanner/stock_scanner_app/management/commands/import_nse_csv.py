import csv
from django.core.management.base import BaseCommand
from stock_scanner_app.models import CompanyInfo

class Command(BaseCommand):
    help = 'Imports NSE data from a CSV file'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='Path to the NSE CSV file')

    def handle(self, *args, **options):
        csv_file = options['csv_file']

        with open(csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                company_info = CompanyInfo(
                    symbol=row['Symbol'],
                    exchange=row['Exchange'],
                    sector=row['Sector'],
                    company_name=row['Company name'],
                    country=row['Country']
                )
                company_info.save()

        self.stdout.write(self.style.SUCCESS('Successfully imported NSE data from CSV file'))