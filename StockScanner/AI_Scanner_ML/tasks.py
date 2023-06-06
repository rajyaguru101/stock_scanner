import yfinance as yf
from datetime import datetime
from AI_Scanner_ML.models import StockData
from AI_Scanner_ML.nse_symbols import get_nifty500_symbols

def download_and_store_stock_data(symbol, start_date, end_date):
    stock = yf.download(symbol + '.NS', start=start_date, end=end_date)

    for index, row in stock.iterrows():
        stock_data = StockData(
            date=index,
            symbol=symbol,
            open=row['Open'],
            high=row['High'],
            low=row['Low'],
            close=row['Close'],
            volume=row['Volume']
        )
        stock_data.save()



def download_and_store_nse_data():
    symbols = get_nifty500_symbols()
    current_year = datetime.now().year
    start_date = f'{current_year}-01-01'
    end_date = f'{current_year}-12-31'

    for symbol in symbols:
        print(f'Downloading data for {symbol}')
        download_and_store_stock_data(symbol, start_date, end_date)
