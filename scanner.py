import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

TIMEFRAME = "3m"
LIMIT = 120
TOP_SYMBOLS = 80

LONG_SCORE = 70
SHORT_SCORE = 70

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

session = requests.Session()
session.headers.update({"User-Agent": "dusikappa-radar-v10"})


def now():
    return datetime.utcnow().strftime("%m-%d %H:%M:%S")


def send_telegram(msg):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(msg)
        return

    url = "https://api.telegram.org/bot{}/sendMessage".format(TELEGRAM_BOT_TOKEN)
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        resp = session.post(url, data=payload, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print("텔레그램 전송 실패:", e)
        print(msg)


def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def get_json(url, params=None, timeout=15):
    resp = session.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# =========================
# Binance REST
# =========================
def get_symbols():
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"

    data = get_json(url)

    if not isinstance(data, dict):
        raise RuntimeError("exchangeInfo 응답 형식이 dict가 아닙니다.")

    if "symbols" not in data:
        raise RuntimeError("exchangeInfo 응답에 symbols가 없습니다: {}".format(str(data)[:500]))

    symbols = []

    for s in data["symbols"]:
        try:
            if s.get("quoteAsset") != "USDT":
                continue
            if s.get("contractType") != "PERPETUAL":
                continue
            if s.get("status") != "TRADING":
                continue
            symbols.append(s["symbol"])
        except Exception:
            continue

    if not symbols:
        raise RuntimeError("사용 가능한 USDT perpetual 심볼을 찾지 못했습니다.")

    return symbols


def fetch_ohlcv(symbol):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": TIMEFRAME,
        "limit": LIMIT,
    }

    data = get_json(url, params=params)

    if not isinstance(data, list):
        raise RuntimeError("klines 응답 오류 {}: {}".format(symbol, str(data)[:300]))

    if len(data) == 0:
        raise RuntimeError("빈 klines 응답: {}".format(symbol))

    df = pd.DataFrame(data)
    df = df.iloc[:, 0:6]
    df.columns = ["ts", "open", "high", "low", "close", "volume"]

    for c in ["ts", "open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna().reset_index(drop=True)


def fetch_open_interest_hist(symbol, limit_count=4):
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    params = {
        "symbol": symbol,
        "period": "5m",
        "limit": limit_count,
    }

    data = get_json(url, params=params)

    if not isinstance(data, list):
        return []

    vals = []
    for row in data:
        vals.append(safe_float(row.get("sumOpenInterest")))
    return vals


# =========================
# Indicators
# =========================
def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calc_smi(df, length=10):
    hh = df["high"].rolling(length).max()
    ll = df["low"].rolling(length).min()
    mid = (hh + ll) / 2.0

    diff = df["close"] - mid
    rng = hh - ll

    diff_ema = diff.ewm(span=3, adjust=False).mean().ewm(span=3, adjust=False).mean()
    rng_ema = rng.ewm(span=3, adjust=False).mean().ewm(span=3, adjust=False).mean()

    smi = 100.0 * diff_ema / (rng_ema / 2.0).replace(0, np.nan)
    smi_d = smi.ewm(span=3, adjust=False).mean()

    out = df.copy()
    out["smi_k"] = smi.fillna(0)
    out["smi_d"] = smi_d.fillna(0)
    return out


# =========================
# Scoring
# =========================
def score_symbol(symbol):
    try:
        df = fetch_ohlcv(symbol)
    except Exception as e:
        print("OHLCV 실패:", symbol, e)
        return None

    if len(df) < 50:
        return None

    df["rsi"] = calc_rsi(df["close"])
    df = calc_smi(df)

    rsi = safe_float(df["rsi"].iloc[-1])
    rsi_prev = safe_float(df["rsi"].iloc[-2])

    smi_k = safe_float(df["smi_k"].iloc[-1])
    smi_d = safe_float(df["smi_d"].iloc[-1])

    vol_base = safe_float(df["volume"].tail(20).mean(), 1e-9)
    vol_ratio = safe_float(df["volume"].iloc[-1]) / max(vol_base, 1e-9)

    price = safe_float(df["close"].iloc[-1])
    price_prev = safe_float(df["close"].iloc[-2], 1e-9)
    price_change = (price - price_prev) / max(price_prev, 1e-9) * 100.0

    oi_vals = fetch_open_interest_hist(symbol, 4)
    oi_accel = False
    oi_pct = 0.0

    if len(oi_vals) >= 4:
        d1 = oi_vals[-1] - oi_vals[-2]
        d2 = oi_vals[-2] - oi_vals[-3]
        d3 = oi_vals[-3] - oi_vals[-4]
        oi_accel = d1 > d2 > d3 and d1 > 0
        if oi_vals[-2] != 0:
            oi_pct = (oi_vals[-1] - oi_vals[-2]) / oi_vals[-2] * 100.0

    score_long = 0
    score_short = 0

    if vol_ratio > 1.3:
        score_long += 20
        score_short += 20

    if oi_accel:
        score_long += 15
        score_short += 15

    if rsi > rsi_prev:
        score_long += 15
    if rsi < rsi_prev:
        score_short += 15

    if smi_k > smi_d:
        score_long += 20
    if smi_k < smi_d:
        score_short += 20

    if price_change > 0:
        score_long += 10
    if price_change < 0:
        score_short += 10

    return {
        "symbol": symbol,
        "long": score_long,
        "short": score_short,
        "price": price_change,
        "vol": vol_ratio,
        "rsi": rsi,
        "smi_k": smi_k,
        "smi_d": smi_d,
        "oi_accel": oi_accel,
        "oi_pct": oi_pct,
    }


def long_msg(item):
    return (
        "🚀 <b>롱 신호</b>\n"
        "⏰ {}\n\n"
        "{}\n"
        "score {}\n"
        "price {:+.2f}%\n"
        "volume {:.2f}x\n"
        "OI {:+.2f}% {}\n"
        "RSI {:.1f}\n"
        "SMI {:.1f}/{:.1f}"
    ).format(
        now(),
        item["symbol"],
        item["long"],
        item["price"],
        item["vol"],
        item["oi_pct"],
        "(accel)" if item["oi_accel"] else "",
        item["rsi"],
        item["smi_k"],
        item["smi_d"],
    )


def short_msg(item):
    return (
        "💥 <b>숏 신호</b>\n"
        "⏰ {}\n\n"
        "{}\n"
        "score {}\n"
        "price {:+.2f}%\n"
        "volume {:.2f}x\n"
        "OI {:+.2f}% {}\n"
        "RSI {:.1f}\n"
        "SMI {:.1f}/{:.1f}"
    ).format(
        now(),
        item["symbol"],
        item["short"],
        item["price"],
        item["vol"],
        item["oi_pct"],
        "(accel)" if item["oi_accel"] else "",
        item["rsi"],
        item["smi_k"],
        item["smi_d"],
    )


def scan():
    symbols = get_symbols()
    print("심볼 수:", len(symbols))

    results = []

    for s in symbols[:TOP_SYMBOLS]:
        item = score_symbol(s)
        if item is not None:
            results.append(item)
        time.sleep(0.03)

    best_long = sorted(results, key=lambda x: x["long"], reverse=True)[:10]
    best_short = sorted(results, key=lambda x: x["short"], reverse=True)[:10]

    print("롱 후보:", [x["symbol"] for x in best_long])
    print("숏 후보:", [x["symbol"] for x in best_short])

    for item in best_long:
        if item["long"] >= LONG_SCORE:
            send_telegram(long_msg(item))

    for item in best_short:
        if item["short"] >= SHORT_SCORE:
            send_telegram(short_msg(item))


def main():
    try:
        send_telegram("✅ 두식이아빠 코인 레이더 시작\n{}".format(now()))
        scan()
        print("scan done")
    except Exception as e:
        err = "❌ scanner error\n{}\n{}".format(now(), str(e)[:1000])
        print(err)
        send_telegram(err)
        raise


if __name__ == "__main__":
    main()
