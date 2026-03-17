"""
RSI-indikator med MetaTrader 5
Kräver: pip install MetaTrader5 pandas matplotlib
"""

import MetaTrader5 as mt5
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


# ── RSI-beräkning (Wilders smoothing) ─────────────────────────────────────────

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ── Hämta data från MT5 ────────────────────────────────────────────────────────

def fetch_mt5_data(
    symbol: str,
    timeframe,
    n_bars: int = 500,
    login: int = None,
    password: str = None,
    server: str = None,
) -> pd.DataFrame:
    """
    Ansluter till MT5 och hämtar OHLCV-data.

    Args:
        symbol:    Instrument, t.ex. "EURUSD", "XAUUSD", "US500"
        timeframe: mt5.TIMEFRAME_H1, TIMEFRAME_D1, TIMEFRAME_M15 osv.
        n_bars:    Antal stängda ljusstakar att hämta
        login:     MT5-kontonummer (None = använd redan inloggad terminal)
        password:  Lösenord (None = använd redan inloggad terminal)
        server:    Mäklarserver (None = använd redan inloggad terminal)
    """
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize misslyckades: {mt5.last_error()}")

    info = mt5.symbol_info(symbol)
    if info is None:
        mt5.shutdown()
        raise ValueError(f"Symbol '{symbol}' hittades inte i MT5.")

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError(f"Ingen data returnerades för {symbol}.")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.rename(columns={"open": "Open", "high": "High",
                        "low": "Low", "close": "Close",
                        "tick_volume": "Volume"}, inplace=True)
    return df[["Open", "High", "Low", "Close", "Volume"]]


# ── Plottning ──────────────────────────────────────────────────────────────────

def plot_rsi(df: pd.DataFrame, symbol: str, timeframe_label: str,
             period: int = 14, overbought: float = 70, oversold: float = 30):

    df["RSI"] = calculate_rsi(df["Close"], period)

    last_rsi = df["RSI"].dropna().iloc[-1]
    last_close = df["Close"].iloc[-1]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [2, 1]},
        sharex=True
    )
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # Prischart
    ax1.plot(df.index, df["Close"], color="#58a6ff", linewidth=1.2, label="Stängningspris")
    ax1.set_ylabel("Pris", color="#8b949e", fontsize=10)
    ax1.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9", fontsize=9)
    ax1.set_title(
        f"{symbol}  |  {timeframe_label}  |  Senaste: {last_close:.5f}",
        color="#c9d1d9", fontsize=12, pad=10
    )
    ax1.grid(color="#21262d", linewidth=0.5)

    # RSI-chart
    rsi_color = "#3fb950" if last_rsi <= oversold else ("#f85149" if last_rsi >= overbought else "#58a6ff")
    ax2.plot(df.index, df["RSI"], color=rsi_color, linewidth=1.2)
    ax2.axhline(overbought, color="#f85149", linewidth=0.8, linestyle="--", alpha=0.7)
    ax2.axhline(oversold,   color="#3fb950", linewidth=0.8, linestyle="--", alpha=0.7)
    ax2.axhline(50,         color="#8b949e", linewidth=0.5, linestyle=":",  alpha=0.5)
    ax2.fill_between(df.index, df["RSI"], overbought,
                     where=(df["RSI"] >= overbought), color="#f85149", alpha=0.15)
    ax2.fill_between(df.index, df["RSI"], oversold,
                     where=(df["RSI"] <= oversold),   color="#3fb950", alpha=0.15)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([oversold, 50, overbought])
    ax2.set_ylabel(f"RSI ({period})", color="#8b949e", fontsize=10)
    ax2.grid(color="#21262d", linewidth=0.5)

    # Signal-etikett
    if last_rsi >= overbought:
        signal = f"ÖVERKÖPT  {last_rsi:.1f}"
        sc = "#f85149"
    elif last_rsi <= oversold:
        signal = f"ÖVERSÅLD  {last_rsi:.1f}"
        sc = "#3fb950"
    else:
        signal = f"NEUTRAL  {last_rsi:.1f}"
        sc = "#8b949e"
    ax2.text(0.01, 0.88, signal, transform=ax2.transAxes,
             color=sc, fontsize=10, fontweight="bold",
             bbox=dict(facecolor="#161b22", edgecolor=sc, boxstyle="round,pad=0.3"))

    # X-axel datum
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    plt.savefig("rsi_mt5.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("Graf sparad: rsi_mt5.png")
    plt.show()


# ── Huvud ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Anpassa dessa inställningar ────────────────────────────────────────────
    SYMBOL     = "XAUUSD"
    TIMEFRAME  = mt5.TIMEFRAME_H1    # H1, H4, D1, M15, M5 osv.
    N_BARS     = 300                 # Antal ljusstakar
    RSI_PERIOD = 14
    OVERBOUGHT = 70
    OVERSOLD   = 30

    # Sätt login/password/server om MT5-terminalen inte redan är inloggad.
    # Lämna None för att använda aktiv session.
    LOGIN    = None   # t.ex. 123456789
    PASSWORD = None   # t.ex. "MittLösenord"
    SERVER   = None   # t.ex. "ICMarkets-Live"

    TIMEFRAME_LABELS = {
        mt5.TIMEFRAME_M1:  "M1",  mt5.TIMEFRAME_M5:  "M5",
        mt5.TIMEFRAME_M15: "M15", mt5.TIMEFRAME_M30: "M30",
        mt5.TIMEFRAME_H1:  "H1",  mt5.TIMEFRAME_H4:  "H4",
        mt5.TIMEFRAME_D1:  "D1",  mt5.TIMEFRAME_W1:  "W1",
    }
    tf_label = TIMEFRAME_LABELS.get(TIMEFRAME, str(TIMEFRAME))

    print(f"Hämtar {N_BARS} stänger för {SYMBOL} ({tf_label}) från MT5...")
    df = fetch_mt5_data(SYMBOL, TIMEFRAME, N_BARS, LOGIN, PASSWORD, SERVER)

    print(f"Senaste stängning : {df['Close'].iloc[-1]:.5f}")
    print(f"Tidsintervall     : {df.index[0]}  →  {df.index[-1]}")

    plot_rsi(df, SYMBOL, tf_label, RSI_PERIOD, OVERBOUGHT, OVERSOLD)

    # Skriv ut de 10 senaste RSI-värdena
    df["RSI"] = calculate_rsi(df["Close"], RSI_PERIOD)
    print("\n── Senaste 10 RSI-värden ──")
    print(df[["Close", "RSI"]].tail(10).to_string())
