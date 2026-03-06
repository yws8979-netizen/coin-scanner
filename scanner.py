import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime

TIMEFRAME = "3m"
LIMIT = 120

TOP_SYMBOLS = 80

LONG_SCORE = 70
SHORT_SCORE = 70

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

session = requests.Session()


def now():
    return datetime.utcnow().strftime("%m-%d %H:%M:%S")


def send_telegram(msg):

    if TELEGRAM_BOT_TOKEN == "" or TELEGRAM_CHAT_ID == "":
        print(msg)
        return

    url = "https://api.telegram.org/bot{}/sendMessage".format(
        TELEGRAM_BOT_TOKEN
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "parse_mode": "HTML"
    }

    try:
        session.post(url, data=payload, timeout=10)
    except:
        pass


# =========================
# 심볼 조회
# =========================

def get_symbols():

    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"

    data = requests.get(url).json()

    symbols = []

    for s in data["symbols"]:

        if s["quoteAsset"] != "USDT":
            continue

        if s["contractType"] != "PERPETUAL":
            continue

        symbols.append(s["symbol"])

    return symbols


# =========================
# KLINE
# =========================

def fetch_ohlcv(symbol):

    url = "https://fapi.binance.com/fapi/v1/klines"

    params = {
        "symbol": symbol,
        "interval": TIMEFRAME,
        "limit": LIMIT
    }

    data = requests.get(url, params=params).json()

    df = pd.DataFrame(data)

    df = df.iloc[:, 0:6]

    df.columns = [
        "ts", "open", "high", "low", "close", "volume"
    ]

    df = df.astype(float)

    return df


# =========================
# RSI
# =========================

def calc_rsi(close, period=14):

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


# =========================
# SMI
# =========================

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


# =========================
# 점수 계산
# =========================

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

        price_change = (price - price_prev) / price_prev * 100

        return {
            "symbol": symbol,
            "long": score_long,
            "short": score_short,
            "price": price_change,
            "vol": vol_ratio,
            "rsi": rsi,
            "smi_k": smi_k,
            "smi_d": smi_d
        }

    except:
        return None


# =========================
# 메시지
# =========================

def long_msg(item):

    return (
        "🚀 <b>롱 신호</b>\n"
        "⏰ {}\n\n"
        "{}\n"
        "score {}\n"
        "price {:+.2f}%\n"
        "volume {:.2f}x\n"
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


def short_msg(item):

    return (
        "💥 <b>숏 신호</b>\n"
        "⏰ {}\n\n"
        "{}\n"
        "score {}\n"
        "price {:+.2f}%\n"
        "volume {:.2f}x\n"
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


# =========================
# 스캔
# =========================

def scan():

    symbols = get_symbols()

    results = []

    for s in symbols[:TOP_SYMBOLS]:

        item = score_symbol(s)

        if item:
            results.append(item)

        time.sleep(0.03)

    best_long = sorted(results, key=lambda x: x["long"], reverse=True)[:10]
    best_short = sorted(results, key=lambda x: x["short"], reverse=True)[:10]

    for item in best_long:

        if item["long"] >= LONG_SCORE:

            send_telegram(long_msg(item))

    for item in best_short:

        if item["short"] >= SHORT_SCORE:

            send_telegram(short_msg(item))


# =========================
# MAIN
# =========================

def main():

    send_telegram("두식이아빠 코인 레이더 시작\n{}".format(now()))

    scan()

    print("scan done")


if __name__ == "__main__":
    main()
