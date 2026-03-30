import MetaTrader5 as mt5
import schedule
import time
import logging

from data_pipeline_propio_v2 import (
    get_data,
    prepare_data,
    create_features,
    predict_next,
    place_order,
    close_old_positions
)

logging.basicConfig(
    filename="live_trader_v2.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    encoding="utf-8"
)

def run_strategy():
    from datetime import datetime
    if datetime.now().weekday() >= 5:
        logging.info("Marknaden stängd – hoppar över.")
        print("Marknaden stängd – hoppar över.")
        return

    logging.info("[SMA20/50 RF] --- Kör strategi ---")
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
    logging.info("[SMA20/50 RF] Live trader startad.")
    print("Startar live trading (V2 - Random Forest)...")
    run_strategy()

    schedule.every().hour.at(":01").do(run_strategy)
    print("Tryck Ctrl+C för att avsluta.")

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        logging.info("[SMA20/50 RF] Live trader avslutad av användaren.")
        print("\nLive trader avslutad.")
        mt5.shutdown()
    except Exception as e:
        logging.error(f"Oväntat fel: {e}")
        print(f"\nOväntat fel: {e}")
        mt5.shutdown()
