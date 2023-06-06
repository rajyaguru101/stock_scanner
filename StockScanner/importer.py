import csv
from ExchangeSelector.models import Stocks

def import_stocks_data(filepath):
    with open(filepath, 'r', encoding='ISO-8859-1') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        counter = 0
        for row in reader:
            symbol = row[0]
            exchange = row[1]
            sector = row[2]
            company = row[3]
            country = row[4]
            
            # Check if the stock already exists in the database
            if not Stocks.objects.filter(symbol=symbol).exists():
                # Create a new stock object
                stock = Stocks(symbol=symbol, exchange=exchange, sector=sector, company=company, country=country)
                # Save the object to the database
                stock.save()
                counter += 1
                print(f"{counter} new stocks added to the database.")
        
        print(f"Total {counter} new stocks added to the database.")
