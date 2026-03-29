import MetaTrader5 as mt5
import schedule
import time
import logging

from data_pipeline_propio import (
    get_data,
    prepare_data,
    create_features,
    predict_next,
    place_order
)

def run_strategy():
    logging.info("--- Kör strategi ---")
    print("\n--- Kör strategi ---")
    try:
        prediction, confidence = predict_next(mode="live")
        if prediction is None:
            logging.warning("Ingen förutsägelse – hoppar över.")
            return
        place_order(prediction, confidence)
    except Exception as e:
        logging.error(f"Fel i run_strategy: {e}")
        print(f"Fel: {e}")

if __name__ == "__main__":
    logging.info("Live trader startad.")
    print("Startar live trading...")
    run_strategy()

    schedule.every().hour.at(":01").do(run_strategy)
    print("Tryck Ctrl+C för att avsluta.")

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        logging.info("Live trader avslutad av användaren.")
        print("\nLive trader avslutad.")
        mt5.shutdown()
    except Exception as e:
        logging.error(f"Oväntat fel i huvudloopen: {e}")
        print(f"\nOväntat fel: {e}")
        mt5.shutdown()