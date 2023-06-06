import requests
from bs4 import BeautifulSoup

def get_nifty500_symbols():
    nifty500_url = 'https://www.niftyindices.com/IndexConstituents/sectorial/6'
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }

    response = requests.get(nifty500_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'csv-data-table'})

    symbols = []
    for row in table.findAll('tr')[1:]:
        symbol = row.findAll('td')[2].text
        symbols.append(symbol)

    return symbols
