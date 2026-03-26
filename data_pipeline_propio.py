
import MetaTrader5 as mt5
from numpy import int64
import pandas as pd
from sklearn.pipeline import Pipeline
import ta
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

symbol = "XAUUSD"  # Valutaparet du vill hämta data för
timeframe = mt5.TIMEFRAME_H1  # Tidsramen du vill häm
start_pos = 0  # Startpositionen för att hämta data
count = 5000  # Antal datapunkter att hämta
rows = 5  # Antal rader att visa i DataFrame
data_frame = pd.DataFrame()  # Skapa en tom DataFrame för att lagra data
forward_hours = 24  # Antal timmar framåt för att skapa målvariabeln
#max_spread = 20  # Maximal spread i punkter för att filtrera bort datapunkter med hög spread

def get_data(bars=count):

    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return  # Avslutar programmet om anslutningen misslyckas

    print("Connected to MetaTrader 5")  # Bekräftar anslutningen
    print(mt5.terminal_info())  # Skriver ut terminalinfo

    data = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, bars)  # Hämtar data från MetaTrader 5

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

    sma50 = data_frame["close"].rolling(window=50).mean()  # Beräknar 50-perioders glidande medelvärde
    sma200 = data_frame["close"].rolling(window=200).mean()  # Beräknar 200-perioders glidande medelvärde
    #data_frame["sma_50"] = (data_frame["close"] - sma50) / sma50  # Skapar en ny kolumn för den normaliserade skillnaden mellan stängningspriset och det glidande medelvärdet
    #data_frame["sma_200"] = (data_frame["close"] - sma200) / sma200  # Skapar en ny kolumn för den normaliserade skillnaden mellan stängningspriset och det glidande medelvärdet
    data_frame["price_to_sma200"] = (data_frame["close"] - sma200) / sma200  # Skapar en ny kolumn för priset i förhållande till SMA200
    data_frame["sma_cross"] = (sma50 - sma200) / sma200  # Skapar en ny kolumn för skillnaden mellan SMA50 och SMA200
    data_frame["rsi"] = ta.momentum.RSIIndicator(data_frame["close"], window=14).rsi()  # Skapar en ny kolumn för RSI-indikatorn
    data_frame["OBV"] = ta.volume.OnBalanceVolumeIndicator(data_frame["close"], data_frame["tick_volume"].astype(int64)).on_balance_volume()  # Skapar en ny kolumn för OBV-indikatorn
    data_frame.dropna(inplace=True)  # Tar bort rader med NaN-värden
    print(data_frame.head(rows))  # Visar de första 5 raderna av DataFrame med nya funktioner
    return data_frame  # Returnerar DataFrame med nya funktioner

def create_labels(data_frame):
    cond1 = data_frame["price_to_sma200"] > 0          # Pris över SMA200
    cond2 = data_frame["sma_cross"] > 0   # SMA50 över SMA200
    cond3 = (data_frame["rsi"] >= 35) & (data_frame["rsi"] <= 65)  # RSI neutralt
    cond4 = data_frame["OBV"].diff() > 0        # OBV stiger

    data_frame["target"] = (cond1 & cond2 & cond3 & cond4).astype(int)
    print(data_frame["target"].value_counts())
    return data_frame

def split_data(data_frame):
    X = data_frame[["price_to_sma200", "sma_cross", "rsi", "OBV"]]  # Funktioner
    Y = data_frame["target"]  # Målvariabel
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False) # Delar upp data i tränings- och testset utan att blanda ordningen
    print(X_train.shape, X_test.shape)
    return X_train, X_test, Y_train, Y_test  # Delar upp data i tränings- och testset

def train_model(X_train, X_test, Y_train):
    scaler = StandardScaler()  # Skapar en standard scaler
    X_train_scaled = scaler.fit_transform(X_train)  # Skalar träningsdata
    X_test_scaled = scaler.transform(X_test)  # Skalar testdata

    model = LogisticRegression(class_weight="balanced")  # Skapar en logistisk regressionsmodell
    #model = RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42)  # Skapar en Random Forest-klassificerare
    model.fit(X_train_scaled, Y_train)  # Tränar modellen på träningsdata
    print("Model trained successfully")  # Bekräftar att modellen har tränats
    joblib.dump(model, "logistic_model.pkl")  # Sparar den tränade modellen till en fil
    joblib.dump(scaler, "scaler.pkl")  # Sparar scalern till en fil
    return model, scaler, X_test_scaled  # Returnerar den tränade modellen och skalad testdata

def cross_validate_model(X, Y):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced"))
    ])
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(pipe, X, Y, cv=tscv, scoring="accuracy")
    print(f"Accuracy per fold: {scores}")
    print(f"Medelvärde: {scores.mean():.3f}")
    print(f"Standardavvikelse: {scores.std():.3f}")

def evaluate_model(model, X_test_scaled, Y_test):
    Y_pred = model.predict(X_test_scaled)
    print(accuracy_score(Y_test, Y_pred))
    print(confusion_matrix(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred, zero_division=0))
    
    feature_names = ["sma_50", "sma_200", "rsi", "OBV"]
    coefficients = model.coef_[0]
    for name, coef in zip(feature_names, coefficients):
        print(f"{name}: {coef:.3f}")

def predict_next(): # Funktion för att göra en förutsägelse baserat på en redan tränad modell
    model = joblib.load("logistic_model.pkl")  # Laddar den tränade modellen
    scaler = joblib.load("scaler.pkl")  # Laddar scalern
    count_new_data = 300    
    data_frame = get_data(count_new_data)
    data_frame = prepare_data(data_frame)
    data_frame = create_features(data_frame)
    latest = data_frame[["price_to_sma200", "sma_cross", "rsi", "OBV"]].iloc[-1:]
    latest_scaled = scaler.transform(latest)
    prediction = model.predict(latest_scaled)
    probability = model.predict_proba(latest_scaled)
    direction = "UPP" if prediction[0] == 1 else "NER"
    confidence = probability[0][prediction[0]] * 100
    print(f"Förutsägelse: {direction} ({confidence:.1f}% konfidens)")

def filter_data(data_frame):
    rows_before = len(data_frame)
    data_frame = data_frame[data_frame.index.dayofweek < 5]  # Filtrerar bort helgdagar
    #data_frame = data_frame[data_frame["spread"] <= max_spread]  # Filtrerar bort datapunkter med hög spread
    rows_after = len(data_frame)
    print(f"Data points removed by filtering: {rows_before - rows_after}")
    return data_frame  # Returnerar den filtrerade DataFrame

if __name__ == "__main__":
    print("Starting data pipeline...")
    data_frame = get_data(count)
    data_frame = prepare_data(data_frame)
    #data_frame = filter_data(data_frame)
    data_frame = create_features(data_frame)
    data_frame = create_labels(data_frame)
    X_train, X_test, Y_train, Y_test = split_data(data_frame)
    model, scaler, X_test_scaled = train_model(X_train, X_test, Y_train)
    evaluate_model(model, X_test_scaled, Y_test)
    cross_validate_model(X_train, Y_train)
    print("Data retrieval completed.")
    print("Data pipeline completed.")
    predict_next()