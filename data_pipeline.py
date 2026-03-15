# Fetch and save data
# Connects to MT%, fetches data, calculates indicators, saves CSV


import MetaTrader5 as mt5
import pandas as pd
import ta

def get_data():
    # Connect to MetaTrader 5
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()  

    print("Connected to MetaTrader 5")

    # Fetch XAU/USD data - last 5000 candles
    rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_H1, 0, 5000)

    # Convert to pandas dataframe
    df = pd.DataFrame(rates)

    # Convert time column to readable format
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Add technical indicators
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['SMA_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
    df['SMA_200'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['tick_volume']).on_balance_volume()

    # Disconect from MetaTrader 5
    mt5.shutdown()
    return df