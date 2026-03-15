# Runs everything

import MetaTrader5 as mt5
import pandas as pd
import ta

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

print(df[['time', 'close', 'RSI', 'SMA_50', 'SMA_200', 'OBV']].tail(10))

#TODO: Delete?
#print(df.head())
#print(f"Total rows: {len(df)}")
#print(df.columns.tolist())
#print(df.head(10))


# Disconect from MetaTrader 5
mt5.shutdown()