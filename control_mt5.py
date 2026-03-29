import MetaTrader5 as mt5

mt5.initialize()

# Kolla kontoinformation
account = mt5.account_info()
print(f"Konto: {account.login}")
print(f"Mäklare: {account.company}")
print(f"Balans: {account.balance}")
print(f"Valuta: {account.currency}")

# Kolla om handel är tillåten
terminal = mt5.terminal_info()
print(f"Handel tillåten: {terminal.trade_allowed}")

# Kolla symbolen XAUUSD
symbol_info = mt5.symbol_info("XAUUSD")
if symbol_info:
    print(f"XAUUSD synlig: {symbol_info.visible}")
    print(f"Minsta lot: {symbol_info.volume_min}")
    print(f"Ask: {mt5.symbol_info_tick('XAUUSD').ask}")
    print(f"Bid: {mt5.symbol_info_tick('XAUUSD').bid}")
else:
    print("XAUUSD hittades inte")

mt5.shutdown()