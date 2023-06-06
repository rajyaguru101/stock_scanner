import os
import django
from django.conf import settings

# Set the DJANGO_SETTINGS_MODULE environment variable
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

# Now import the model
from ExchangeSectorSelector.models import Symbols
import csv
from ExchangeSectorSelector.models import Symbols

csv_filepath = 'Symbols.csv'

with open(csv_filepath, 'r') as f:
    reader = csv.reader(f)
    # rest of your code here

def import_data():
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)

        # skip the header row
        next(reader)

        # loop through each row and create a new Symbols object
        for row in reader:
            symbol = row[0]
            exchange = row[1]
            sector = row[2]
            company = row[3]
            country = row[4]

            symbol_obj = Symbols(symbols=symbol, exchanges=exchange, sectors=sector, companies=company, countries=country)
            symbol_obj.save()
            
if __name__ == '__main__':
    import_data()
