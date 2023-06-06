# patterns/features/doji_features.py
import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'StockScanner.settings')
django.setup()
import pandas as pd
import numpy as np
from stock_scanner_app.models import HistoricalData
import ta
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.volume import ChaikinMoneyFlowIndicator
from ta.trend import AroonIndicator
from ta.trend import ADXIndicator
from ta.volume import ChaikinMoneyFlowIndicator
from ta.momentum import StochasticOscillator


from sklearn.preprocessing import MinMaxScaler

def load_data():
    pd.set_option('display.max_columns', None)
    data = HistoricalData.objects.values()
    df = pd.DataFrame.from_records(data)

    # Sort the DataFrame by 'symbol' and 'date'
    df = df.sort_values(by=['symbol', 'date'])

    # Set 'date' as the index
    df.set_index('date', inplace=True)

    # Create a dictionary to store data for each unique symbol
    symbol_data = {}

    # Get the list of unique symbols
    symbols = df['symbol'].unique()

    # Iterate through the unique symbols and store their data in the dictionary
    for symbol in symbols:
        symbol_data[symbol] = df[df['symbol'] == symbol]
    print(symbol_data)
    return 


def identify_dojis(df, tolerance=0.005):
    doji_mask = abs(df['open'] - df['close']) <= (tolerance * (df['high'] - df['low']))
    return doji_mask

def label_trend_reversals(df, reversal_threshold=0.03):
    df['future_price'] = df.groupby('symbol')['close'].shift(-1)
    df['price_change'] = (df['future_price'] - df['close']) / df['close']
    df['reversal'] = np.where(abs(df['price_change']) >= reversal_threshold, 1, 0)
    return df

def generate_features(df):
    # Calculate moving averages
    df['SMA_5'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=5).mean())
    df['SMA_10'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=10).mean())

    # Calculate RSI
    delta = df.groupby('symbol')['close'].apply(lambda x: x.diff())
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Calculate historical volatility
    df['log_return'] = df.groupby('symbol')['close'].apply(lambda x: np.log(x) - np.log(x.shift(1)))
    df['volatility'] = df.groupby('symbol')['log_return'].apply(lambda x: x.rolling(window=10).std())

    # Calculate Bollinger Bands
    df['bollinger_upper'] = df.groupby('symbol')['close'].transform(lambda x: BollingerBands(x, window=20, window_dev=2).bollinger_hband())
    df['bollinger_lower'] = df.groupby('symbol')['close'].transform(lambda x: BollingerBands(x, window=20, window_dev=2).bollinger_lband())

    # Calculate MACD
    df['macd'] = df.groupby('symbol')['close'].transform(lambda x: MACD(x, window_slow=26, window_fast=12, window_sign=9).macd())
    df['macd_signal'] = df.groupby('symbol')['close'].transform(lambda x: MACD(x, window_slow=26, window_fast=12, window_sign=9).macd_signal())


    # Add new features
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['body_length'] = abs(df['open'] - df['close'])
    df['shadow_body_ratio'] = np.where(df['body_length'] != 0, (df['upper_shadow'] + df['lower_shadow']) / df['body_length'], df['upper_shadow'] + df['lower_shadow'])

    df['roc'] = df.groupby('symbol')['close'].transform(lambda x: (x - x.shift(10)) / x.shift(10))

    def apply_indicator(group, indicator_func, **kwargs):
        if len(group) >= kwargs['window']:
            return indicator_func(**kwargs)
        else:
            return pd.Series(index=group.index, dtype=float)

    df['adx'] = df.groupby('symbol').apply(lambda x: apply_indicator(x, ADXIndicator, high=x['high'], low=x['low'], close=x['close'], window=14).adx()).reset_index(level=0, drop=True)
    df['aroon_up'] = df.groupby('symbol').apply(lambda x: apply_indicator(x, AroonIndicator, close=x['close'], window=25).aroon_up()).reset_index(level=0, drop=True)
    df['aroon_down'] = df.groupby('symbol').apply(lambda x: apply_indicator(x, AroonIndicator, close=x['close'], window=25).aroon_down()).reset_index(level=0, drop=True)
    df['cmf'] = df.groupby('symbol').apply(lambda x: apply_indicator(x, ChaikinMoneyFlowIndicator, high=x['high'], low=x['low'], close=x['close'], volume=x['volume'], window=20).chaikin_money_flow()).reset_index(level=0, drop=True)
    df['stoch_k'] = df.groupby('symbol').apply(lambda x: apply_indicator(x, StochasticOscillator, high=x['high'], low=x['low'], close=x['close'], window=14).stoch()).reset_index(level=0, drop=True)
    df['stoch_d'] = df.groupby('symbol').apply(lambda x: apply_indicator(x, StochasticOscillator, high=x['high'], low=x['low'], close=x['close'], window=14).stoch_signal()).reset_index(level=0, drop=True)
    df = df.dropna()

    return df


def normalize_features(df, feature_columns):
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df

