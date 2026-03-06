import os
import time
import requests
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime

# =========================
# 기본 설정
# =========================

TIMEFRAME = "3m"
OHLCV_LIMIT = 120

TOP_VOLUME_FILTER = 80
TOP_N_ALERTS = 10

LONG_EARLY_THRESHOLD = 60
SHORT_EARLY_THRESHOLD = 60

LONG_CONFIRM_THRESHOLD = 72
SHORT_CONFIRM_THRESHOLD = 72

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

exchange = ccxt.binance({
    "enableRateLimit": True,
    "options": {"defaultType": "future"}
})

session = requests.Session()

# =========================
# 유틸
# =========================

def now_str():
    return datetime.utcnow().strftime("%m-%d %H:%M:%S")


def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default


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
    except Exception as e:
        print("텔레그램 실패", e)


# =========================
# 시장 정보
# =========================

def get_usdt_perp_symbols():

    markets = exchange.load_markets()

    symbols = []

    for sym, m in markets.items():

        if m.get("quote") != "USDT":
            continue

        if not m.get("swap"):
            continue

        if not m.get("active"):
            continue

        symbols.append(sym)

    return symbols


# =========================
# OHLCV
# =========================

def fetch_ohlcv(symbol):

    raw = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=OHLCV_LIMIT)

    df = pd.DataFrame(raw, columns=[
        "ts","open","high","low","close","volume"
    ])

    return df


# =========================
# OI
# =========================

def fetch_open_interest(symbol):

    sym = symbol.replace("/", "").replace(":USDT","")

    url = "https://fapi.binance.com/futures/data/openInterestHist"

    params = {
        "symbol": sym,
        "period": "5m",
        "limit": 4
    }

    r = session.get(url, params=params)

    data = r.json()

    oi = []

    for x in data:
        oi.append(safe_float(x["sumOpenInterest"]))

    return oi


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

    high = df["high"]
    low = df["low"]
    close = df["close"]

    length = 10

    hh = high.rolling(length).max()
    ll = low.rolling(length).min()

    mid = (hh + ll) / 2

    diff = close - mid
    rng = hh - ll

    diff_ema = diff.ewm(span=3).mean().ewm(span=3).mean()
    rng_ema = rng.ewm(span=3).mean().ewm(span=3).mean()

    smi = 100 * diff_ema / (rng_ema / 2)

    smi_d = smi.ewm(span=3).mean()

    df["smi_k"] = smi
    df["smi_d"] = smi_d

    return df


# =========================
# 거래량 상위 필터
# =========================

def get_top_volume_symbols(symbols):

    vols = []

    for s in symbols[:150]:

        try:

            df = fetch_ohlcv(s)

            v = df["volume"].iloc[-1]

            vols.append((s,v))

        except:
            continue

    vols.sort(key=lambda x: x[1], reverse=True)

    return [x[0] for x in vols[:TOP_VOLUME_FILTER]]


# =========================
# 점수 계산
# =========================

def score_symbol(symbol):

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

    oi = fetch_open_interest(symbol)

    oi_accel = False

    if len(oi) >= 4:

        d1 = oi[-1] - oi[-2]
        d2 = oi[-2] - oi[-3]
        d3 = oi[-3] - oi[-4]

        oi_accel = d1 > d2 > d3

    price = df["close"].iloc[-1]
    price_prev = df["close"].iloc[-2]

    price_chg = (price - price_prev) / price_prev * 100

    long_score = 0
    short_score = 0

    if oi_accel:
        long_score += 20
        short_score += 20

    if vol_ratio > 1.3:
        long_score += 15
        short_score += 15

    if rsi > rsi_prev:
        long_score += 10

    if rsi < rsi_prev:
        short_score += 10

    if smi_k > smi_d:
        long_score += 10

    if smi_k < smi_d:
        short_score += 10

    return {
        "symbol": symbol,
        "price_chg": price_chg,
        "long_score": long_score,
        "short_score": short_score,
        "rsi": rsi,
        "smi_k": smi_k,
        "smi_d": smi_d,
        "vol_ratio": vol_ratio,
        "oi_accel": oi_accel
    }


# =========================
# 메시지
# =========================

def make_long_msg(item):

    return (
        "🚀 <b>롱 시그널</b>\n"
        "⏰ {}\n\n"
        "{}\n"
        "score {}\n"
        "price {:+.2f}%\n"
        "vol {:.2f}x\n"
        "RSI {:.1f}\n"
        "SMI {:.1f}/{:.1f}"
    ).format(
        now_str(),
        item["symbol"],
        item["long_score"],
        item["price_chg"],
        item["vol_ratio"],
        item["rsi"],
        item["smi_k"],
        item["smi_d"]
    )


def make_short_msg(item):

    return (
        "💥 <b>숏 시그널</b>\n"
        "⏰ {}\n\n"
        "{}\n"
        "score {}\n"
        "price {:+.2f}%\n"
        "vol {:.2f}x\n"
        "RSI {:.1f}\n"
        "SMI {:.1f}/{:.1f}"
    ).format(
        now_str(),
        item["symbol"],
        item["short_score"],
        item["price_chg"],
        item["vol_ratio"],
        item["rsi"],
        item["smi_k"],
        item["smi_d"]
    )


# =========================
# 스캔
# =========================

def scan():

    symbols = get_usdt_perp_symbols()

    symbols = get_top_volume_symbols(symbols)

    print("scan symbols:", len(symbols))

    scored = []

    for s in symbols:

        try:

            item = score_symbol(s)

            if item:
                scored.append(item)

            time.sleep(0.05)

        except Exception as e:
            print("err", s, e)

    best_long = sorted(scored, key=lambda x: x["long_score"], reverse=True)[:TOP_N_ALERTS]

    best_short = sorted(scored, key=lambda x: x["short_score"], reverse=True)[:TOP_N_ALERTS]

    for item in best_long:

        if item["long_score"] >= LONG_CONFIRM_THRESHOLD:

            send_telegram(make_long_msg(item))

    for item in best_short:

        if item["short_score"] >= SHORT_CONFIRM_THRESHOLD:

            send_telegram(make_short_msg(item))


# =========================
# MAIN
# =========================

def main():

    send_telegram(
        "✅ 두식이아빠 코인탐지기 v8 시작\n{}".format(now_str())
    )

    scan()

    print("done")


if __name__ == "__main__":
    main()
