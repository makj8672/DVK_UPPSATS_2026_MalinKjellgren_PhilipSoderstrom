# Runs everything

from data_pipeline import get_data

df = get_data()
print(df[['time', 'close', 'RSI', 'SMA_50', 'SMA_200', 'OBV']].tail(10))
