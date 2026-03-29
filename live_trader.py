import MetaTrader5 as mt5
import pandas as pd
import joblib
import ta
import numpy as np
import schedule
import time

from data_pipeline_propio import (
    get_data,
    prepare_data,
    create_features,
    predict_next,
    place_order
)

def run_strategy():
    print(f"\n--- Kör strategi ---")
    prediction, confidence = predict_next(mode="live")
    place_order(prediction, confidence)

if __name__ == "__main__":
    print("Startar live trading...")
    run_strategy()  # Kör strategin direkt vid start

    schedule.every().hour.at(":00").do(run_strategy)
    
    while True:
        schedule.run_pending()
        time.sleep(60)