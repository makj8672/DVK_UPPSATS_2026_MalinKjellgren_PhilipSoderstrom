#data_pipeline.py

"""Data pipeline for fetching, processing and preparing data for trading strategy backtests.
Connects to MT%, fetches data, calculates indicators, saves CSV"""
 
import sys
import MetaTrader5 as mt5
import pandas as pd
import ta

SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
CANDLES = 10000
SMA_SHORT = 50
SMA_LONG = 200

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
    - obv_diff: Procentual change in OBV from previous period"""

    sma_short = df['close'].rolling(window=SMA_SHORT).mean()
    sma_long = df['close'].rolling(window=SMA_LONG).mean()
    df["price_to_sma"] = (df['close'] - sma_long) / sma_long
    df["sma_cross"] = (sma_short - sma_long) / sma_long
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

def get_data():
    """Fetch historical data from MT5 and return as a DataFrame."""
    try:
        connect_to_mt5()                                                        # Connect to MT5
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, CANDLES)          # Fetch historical data (for specified symbol, timeframe, number of candles)
        if rates is None or len(rates) == 0:                                    # Check if data was fetched successfully
            raise RuntimeError(f"Could not fetch data: {mt5.last_error()}")     # If no data was fetched, raise error message
        df = pd.DataFrame(rates)                                                # Convert data to DataFrame
        df['time'] = pd.to_datetime(df['time'], unit='s')                       # Convert time column to datetime
        return df                                                               # Return the DataFrame       
    
    except ConnectionError as e:    # Handle potential connection errors
        print(e)
        sys.exit(1)                 # If connection fails -> exit the program with an error code
    finally:
        mt5.shutdown()              # Ensure MT5 connection is closed when done

def split_data(df):
    """Split data chronologically into train, validation and test sets.
    
    - 60% training
    - 20% validation
    - 20% test
    """
    n = len(df)                             # Get total number of rows in the DataFrame
    train_end = int(n * 0.6)                # Calculate index for end of training data (60% of total)
    val_end = int(n * 0.8)                  # Calculate index for end of validation data (80% of total)

    train_data = df.iloc[:train_end]        # Select training data
    val_data = df.iloc[train_end:val_end]   # Select validation data
    test_data = df.iloc[val_end:]           # Select test data

    return train_data, val_data, test_data  # Return the three datasets