import os
import time
import requests
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime

TIMEFRAME = "3m"
OHLCV_LIMIT = 120
TOP_VOLUME_FILTER = 80
TOP_N_ALERTS = 10

LONG_THRESHOLD = 70
SHORT_THRESHOLD = 70

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

exchange = ccxt.binance({
    "enableRateLimit": True,
    "options": {"defaultType": "future"}
})

session = requests.Session()


def now():
    return datetime.utcnow().strftime("%m-%d %H:%M:%S")


def send_telegram(msg):

    if TELEGRAM_BOT_TOKEN == "" or TELEGRAM_CHAT_ID == "":
        print(msg)
        return

    url = "https://api.telegram.org/bot{}/sendMessage".format(TELEGRAM_BOT_TOKEN)

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "parse_mode": "HTML"
    }

    try:
        session.post(url, data=payload, timeout=10)
    except:
        pass


def get_symbols():

    markets = exchange.load_markets()

    result = []

    for sym, m in markets.items():

        if m.get("quote") != "USDT":
            continue

        if not m.get("swap"):
            continue

        result.append(sym)

    return result


def fetch_ohlcv(symbol):

    data = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=OHLCV_LIMIT)

    df = pd.DataFrame(data, columns=[
        "ts", "open", "high", "low", "close", "volume"
    ])

    return df


def calc_rsi(close, period=14):

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


def calc_smi(df):

    length = 10

    hh = df["high"].rolling(length).max()
    ll = df["low"].rolling(length).min()

    mid = (hh + ll) / 2

    diff = df["close"] - mid
    rng = hh - ll

    diff_ema = diff.ewm(span=3).mean().ewm(span=3).mean()
    rng_ema = rng.ewm(span=3).mean().ewm(span=3).mean()

    smi = 100 * diff_ema / (rng_ema / 2)
    smi_d = smi.ewm(span=3).mean()

    df["smi_k"] = smi
    df["smi_d"] = smi_d

    return df


def score_symbol(symbol):

    try:

        df = fetch_ohlcv(symbol)

        if len(df) < 50:
            return None

        df["rsi"] = calc_rsi(df["close"])
        df = calc_smi(df)

        rsi = df["rsi"].iloc[-1]
        rsi_prev = df["rsi"].iloc[-2]

        smi_k = df["smi_k"].iloc[-1]
        smi_d = df["smi_d"].iloc[-1]

        vol_ratio = df["volume"].iloc[-1] / df["volume"].tail(20).mean()

        score_long = 0
        score_short = 0

        if vol_ratio > 1.3:
            score_long += 20
            score_short += 20

        if rsi > rsi_prev:
            score_long += 15

        if rsi < rsi_prev:
            score_short += 15

        if smi_k > smi_d:
            score_long += 15

        if smi_k < smi_d:
            score_short += 15

        price = df["close"].iloc[-1]
        price_prev = df["close"].iloc[-2]

        price_chg = (price - price_prev) / price_prev * 100

        return {
            "symbol": symbol,
            "long": score_long,
            "short": score_short,
            "price": price_chg,
            "vol": vol_ratio,
            "rsi": rsi,
            "smi_k": smi_k,
            "smi_d": smi_d
        }

    except:
        return None


def make_long_msg(item):

    msg = (
        "🚀 <b>롱 시그널</b>\n"
        "⏰ {}\n\n"
        "{}\n"
        "score {}\n"
        "price {:+.2f}%\n"
        "vol {:.2f}x\n"
        "RSI {:.1f}\n"
        "SMI {:.1f}/{:.1f}"
    ).format(
        now(),
        item["symbol"],
        item["long"],
        item["price"],
        item["vol"],
        item["rsi"],
        item["smi_k"],
        item["smi_d"]
    )

    return msg


def make_short_msg(item):

    msg = (
        "💥 <b>숏 시그널</b>\n"
        "⏰ {}\n\n"
        "{}\n"
        "score {}\n"
        "price {:+.2f}%\n"
        "vol {:.2f}x\n"
        "RSI {:.1f}\n"
        "SMI {:.1f}/{:.1f}"
    ).format(
        now(),
        item["symbol"],
        item["short"],
        item["price"],
        item["vol"],
        item["rsi"],
        item["smi_k"],
        item["smi_d"]
    )

    return msg


def scan():

    symbols = get_symbols()

    scored = []

    for s in symbols:

        item = score_symbol(s)

        if item:
            scored.append(item)

        time.sleep(0.05)

    best_long = sorted(scored, key=lambda x: x["long"], reverse=True)[:TOP_N_ALERTS]
    best_short = sorted(scored, key=lambda x: x["short"], reverse=True)[:TOP_N_ALERTS]

    for item in best_long:

        if item["long"] >= LONG_THRESHOLD:

            send_telegram(make_long_msg(item))

    for item in best_short:

        if item["short"] >= SHORT_THRESHOLD:

            send_telegram(make_short_msg(item))


def main():

    send_telegram("두식이아빠 코인탐지기 시작\n{}".format(now()))

    scan()

    print("scan complete")


if __name__ == "__main__":
    main()
