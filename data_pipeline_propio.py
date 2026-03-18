
import MetaTrader5 as mt5
import pandas as pd

symbol = "XAUUSD"  # Valutaparet du vill hämta data för
timeframe = mt5.TIMEFRAME_H1  # Tidsramen du vill häm
start_pos = 0  # Startpositionen för att hämta data
count = 1000  # Antal datapunkter att hämta
rows = 5  # Antal rader att visa i DataFrame
data_frame = pd.DataFrame()  # Skapa en tom DataFrame för att lagra data

def get_data():

    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return  # Avslutar programmet om anslutningen misslyckas

    print("Connected to MetaTrader 5")  # Bekräftar anslutningen
    print(mt5.terminal_info())  # Skriver ut terminalinfo

    data = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)  # Hämtar data från MetaTrader 5

    if data is None or len(data) == 0:
        print("No data retrieved, error code =", mt5.last_error())
        return  # Avslutar programmet om datainhämtningen misslyckas

    data_frame  = pd.DataFrame(data) # Skapa en DataFrame från den hämtade datan

    print(data_frame.head(rows))  # Visar de första 5 raderna av DataFrame
    
    mt5.shutdown()  # Avslutar anslutningen
    return  data_frame  # Returnerar DataFrame

def prepare_data(data_frame):
    data_frame["time"] = pd.to_datetime(data_frame["time"], unit="s")  # Konverterar tidsstämplar till datetime
    data_frame.set_index("time", inplace=True)  # Sätter "time" som index
    print(data_frame.describe())  # Visar en beskrivning av DataFrameda
    print(data_frame.dtypes)  # Visar datatyperna i DataFrame
    return data_frame  # Returnerar den förberedda DataFrame


if __name__ == "__main__":
    print("Starting data pipeline...")
    data_frame = get_data()
    prepare_data(data_frame) 
    print("Data retrieval completed.")
    print("Data pipeline completed.")