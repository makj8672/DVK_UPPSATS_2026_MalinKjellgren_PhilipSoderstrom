
import MetaTrader5 as mt5
import pandas as pd
from sklearn.model_selection import train_test_split

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

def create_features(data_frame):
    data_frame["returns"] = data_frame["close"].pct_change()  # Skapar en ny kolumn för avkastning
    data_frame["sma_10"] = data_frame["close"].rolling(window=10).mean()  # Skapar en ny kolumn för 10-perioders glidande medelvärde
    data_frame["volatility"] = data_frame["returns"].rolling(window=10).std()  # Skapar en ny kolumn för volatilitet
    data_frame.dropna(inplace=True)  # Tar bort rader med NaN-värden
    print(data_frame.head(rows))  # Visar de första 5 raderna av DataFrame med nya funktioner
    return data_frame  # Returnerar DataFrame med nya funktioner

def create_labels(data_frame):
    data_frame["target"] = (data_frame["close"].shift(-1) > data_frame["close"]).astype(int)  # Skapar en ny kolumn för målvariabeln
    data_frame.dropna(inplace=True)  # Tar bort rader med NaN-vär
    print(data_frame["target"].value_counts())  # Visar fördelningen av målvariabeln
    return data_frame  # Returnerar DataFrame med målvariabeln

def split_data(data_frame):
    X = data_frame[["returns", "sma_10", "volatility"]]  # Funktioner
    y = data_frame["target"]  # Målvariabel
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) # Delar upp data i tränings- och testset utan att blanda ordningen
    print(X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test  # Delar upp data i tränings- och testset

if __name__ == "__main__":
    print("Starting data pipeline...")
    data_frame = get_data()
    data_frame = prepare_data(data_frame)
    data_frame = create_features(data_frame)
    data_frame = create_labels(data_frame)
    X_train, X_test, y_train, y_test = split_data(data_frame)
    print("Data retrieval completed.")
    print("Data pipeline completed.")