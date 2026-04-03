#data_pipeline.py
# Fetch and save data
# Connects to MT%, fetches data, calculates indicators, saves CSV

# Psudo code:
# 1. Connect to MT5
# 2. Fetch XAU/USD data - last 5000 candles
# 3. Convert to pandas dataframe
# 4. Add and calculate technical indicators (RSI, SMA_50, SMA_200, OBV)
# 5. Create target variable (did price go up next hour? 1=yes, 0=no)
# 6. Remove rows with missing values
# 7. Return dataframe

import sys
import MetaTrader5 as mt5
import pandas as pd
import ta

SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
CANDLES = 5000

def connect_to_mt5():
    if not mt5.initialize():
        raise ConnectionError(f"MT5 initialize() failed: {mt5.last_error()}")
    print("Connected to MetaTrader 5")


def create_features(df):
    """Create features for machine learning model.
    
    Calculates SMA50 and SMA200 internally to derive the following features:
    - price_to_sma200: how far price is from SMA200, as a ratio
    - sma_cross: difference between SMA50 and SMA200, as a ratio
    - rsi: Relative Strength Index (14 periods)
    - obv: On Balance Volume
    - obv_diff: Procentual change in OBV from previous period

    TODO: These features are likely suboptimal, we should experiment with adding or replacing them.
    """
    sma50 = df['close'].rolling(window=50).mean()
    sma200 = df['close'].rolling(window=200).mean()
    df["price_to_sma200"] = (df['close'] - sma200) / sma200
    df["sma_cross"] = (sma50 - sma200) / sma200
    df["rsi"] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['tick_volume']).on_balance_volume()
    df["obv_diff"] = df["obv"].pct_change()
    return df

def create_target(df):
    """Create target variable for machine learning model. 
    We are trying to predict if the price will go up in the next hour."""
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int) # Shift close price by -1 to get next hour's price and create binary target
    return df

def clean_data(df):
    """Remove rows with missing values."""
    df = df.dropna() # Drop rows with missing values
    return df


# def get_data():
    # Connect to MetaTrader 5
    try:
        connect_to_mt5()
    except ConnectionError as e:
        print(e)
        sys.exit(1)

    try:
        # Fetch XAU/USD data - last 5000 candles
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, CANDLES)

        if rates is None or len(rates) == 0:
            raise RuntimeError(f"Kunde inte hämta data: {mt5.last_error()}")
        
        # Convert to pandas dataframe
        df = pd.DataFrame(rates)

        # Convert time column to readable format
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Add technical indicators
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['SMA_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        df['SMA_200'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['tick_volume']).on_balance_volume()

        return df
    finally:
        # Disconect from MetaTrader 5
        mt5.shutdown()

def get_data():
    try:
        connect_to_mt5()
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, CANDLES)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"Kunde inte hämta data: {mt5.last_error()}")
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    except ConnectionError as e:
        print(e)
        sys.exit(1)
    finally:
        mt5.shutdown()