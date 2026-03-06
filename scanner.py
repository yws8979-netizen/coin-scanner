import os
import time
from datetime import datetime

import ccxt
import numpy as np
import pandas as pd
import requests


# =========================
# 기본 설정
# =========================
TIMEFRAME = "3m"
OHLCV_LIMIT = 120
TOP_N_ALERTS = 10

LONG_EARLY_THRESHOLD = 60
SHORT_EARLY_THRESHOLD = 60
LONG_CONFIRM_THRESHOLD = 72
SHORT_CONFIRM_THRESHOLD = 72

PRICE_EARLY_LONG_MIN = 0.2
PRICE_EARLY_LONG_MAX = 1.8
PRICE_EARLY_SHORT_MIN = -1.8
PRICE_EARLY_SHORT_MAX = -0.2

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

last_alert_map = {}

exchange = ccxt.binance(
    {
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    }
)

session = requests.Session()
session.headers.update({"User-Agent": "scanner-v7-safe"})


# =========================
# 유틸
# =========================
def now_str():
    return datetime.now().strftime("%m-%d %H:%M:%S")


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def send_telegram(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[텔레그램 미설정]")
        print(text)
        return

    url = "https://api.telegram.org/bot" + TELEGRAM_BOT_TOKEN + "/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        resp = session.post(url, data=payload, timeout=10)
        resp.raise_for_status()
        print("[텔레그램 전송 성공]")
    except Exception as e:
        print("[텔레그램 전송 실패]", e)
        print(text)


# =========================
# 바이낸스 REST
# =========================
def binance_symbol(symbol):
    return symbol.replace("/", "").replace(":USDT", "")


def get_usdt_perp_symbols():
    markets = exchange.load_markets()
    symbols = []

    for sym, market in markets.items():
        try:
            if not market.get("active", True):
                continue
            if market.get("quote") != "USDT":
                continue
            if not market.get("swap", False):
                continue
            symbols.append(sym)
        except Exception:
            continue

    return sorted(symbols)


def fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=OHLCV_LIMIT):
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna().reset_index(drop=True)


def fetch_open_interest_history(symbol, period="5m", limit=4):
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    params = {
        "symbol": binance_symbol(symbol),
        "period": period,
        "limit": limit,
    }

    r = session.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    result = []
    for row in data:
        result.append(safe_float(row.get("sumOpenInterest")))
    return result


def fetch_taker_ratio(symbol, period="5m", limit=4):
    url = "https://fapi.binance.com/futures/data/takerlongshortRatio"
    params = {
        "symbol": binance_symbol(symbol),
        "period": period,
        "limit": limit,
    }

    r = session.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    parsed = []
    for row in data:
        parsed.append(
            {
                "buySellRatio": safe_float(row.get("buySellRatio")),
                "buyVol": safe_float(row.get("buyVol")),
                "sellVol": safe_float(row.get("sellVol")),
            }
        )
    return parsed


# =========================
# 지표 계산
# =========================
def calc_rsi(close, length=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calc_smi(df, length_k=10, length_d=3, ema_len=3):
    hh = df["high"].rolling(length_k).max()
    ll = df["low"].rolling(length_k).min()
    mid = (hh + ll) / 2.0
    diff = df["close"] - mid
    rng = hh - ll

    diff_ema = diff.ewm(span=ema_len, adjust=False).mean().ewm(span=ema_len, adjust=False).mean()
    rng_ema = rng.ewm(span=ema_len, adjust=False).mean().ewm(span=ema_len, adjust=False).mean()

    smi_k = 100 * (diff_ema / (rng_ema / 2.0).replace(0, np.nan))
    smi_d = smi_k.ewm(span=length_d, adjust=False).mean()

    out = df.copy()
    out["smi_k"] = smi_k.fillna(0)
    out["smi_d"] = smi_d.fillna(0)
    return out


def add_indicators(df):
    out = df.copy()
    out["rsi"] = calc_rsi(out["close"], 14)
    out["ema7"] = out["close"].ewm(span=7, adjust=False).mean()
    out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["candle_range"] = out["high"] - out["low"]
    out["body"] = (out["close"] - out["open"]).abs()
    out = calc_smi(out)
    return out.dropna().reset_index(drop=True)


# =========================
# 조건 함수
# =========================
def pct_change(a, b):
    if b == 0:
        return 0.0
    return (a - b) / b * 100.0


def oi_acceleration(oi_vals):
    if len(oi_vals) < 4:
        return False, 0.0

    d1 = oi_vals[-1] - oi_vals[-2]
    d2 = oi_vals[-2] - oi_vals[-3]
    d3 = oi_vals[-3] - oi_vals[-4]

    accel = d1 > d2 > d3 and d1 > 0
    latest_pct = pct_change(oi_vals[-1], oi_vals[-2])
    return accel, latest_pct


def volume_acceleration(vols):
    if len(vols) < 4:
        return False, 0.0

    v1 = safe_float(vols[-1])
    v2 = safe_float(vols[-2])
    v3 = safe_float(vols[-3])

    r1 = v1 / max(v2, 1e-9)
    r2 = v2 / max(v3, 1e-9)

    accel = r1 > r2 and r1 > 1.12 and v1 > v2 > v3

    if len(vols) >= 20:
        lookback = vols[-20:]
    else:
        lookback = vols

    vol_ratio = v1 / max(np.mean(lookback), 1e-9)
    return accel, vol_ratio


def is_range_tight(df):
    recent = df["candle_range"].tail(5).mean()
    base = df["candle_range"].tail(30).mean()

    if pd.isna(recent) or pd.isna(base) or base == 0:
        return False

    return recent < base * 0.72


def upper_wick_reject(df):
    row = df.iloc[-1]
    upper = row["high"] - max(row["open"], row["close"])
    body = abs(row["close"] - row["open"])
    return upper > body * 1.2 and row["close"] < row["high"]


def lower_wick_reject(df):
    row = df.iloc[-1]
    lower = min(row["open"], row["close"]) - row["low"]
    body = abs(row["close"] - row["open"])
    return lower > body * 1.2 and row["close"] > row["low"]


def breakout_up(df):
    prev_high = df["high"].iloc[-6:-1].max()
    return df["close"].iloc[-1] > prev_high


def breakout_down(df):
    prev_low = df["low"].iloc[-6:-1].min()
    return df["close"].iloc[-1] < prev_low


def rsi_long_early(df):
    r1 = df["rsi"].iloc[-1]
    r2 = df["rsi"].iloc[-2]
    r3 = df["rsi"].iloc[-3]
    return (r1 > r2 > r3) and (r1 < 52)


def rsi_short_early(df):
    r1 = df["rsi"].iloc[-1]
    r2 = df["rsi"].iloc[-2]
    r3 = df["rsi"].iloc[-3]
    return (r1 < r2 < r3) and (r1 > 48)


def smi_long_early(df):
    k1 = df["smi_k"].iloc[-1]
    k2 = df["smi_k"].iloc[-2]
    d1 = df["smi_d"].iloc[-1]
    d2 = df["smi_d"].iloc[-2]

    gap_now = d1 - k1
    gap_prev = d2 - k2
    return k1 > k2 and gap_now < gap_prev


def smi_short_early(df):
    k1 = df["smi_k"].iloc[-1]
    k2 = df["smi_k"].iloc[-2]
    d1 = df["smi_d"].iloc[-1]
    d2 = df["smi_d"].iloc[-2]

    gap_now = k1 - d1
    gap_prev = k2 - d2
    return k1 < k2 and gap_now < gap_prev


def taker_delta_info(taker_rows):
    if len(taker_rows) < 2:
        return 0.0, False, False

    latest = taker_rows[-1]
    prev = taker_rows[-2]

    latest_delta = latest["buyVol"] - latest["sellVol"]
    prev_delta = prev["buyVol"] - prev["sellVol"]

    long_bias = latest_delta > 0 and latest_delta > prev_delta
    short_bias = latest_delta < 0 and latest_delta < prev_delta

    return latest_delta, long_bias, short_bias


# =========================
# 점수 계산
# =========================
def score_symbol(symbol):
    df = fetch_ohlcv(symbol)
    if len(df) < 50:
        return None

    df = add_indicators(df)

    oi_vals = fetch_open_interest_history(symbol, period="5m", limit=4)
    taker_rows = fetch_taker_ratio(symbol, period="5m", limit=4)

    oi_accel, oi_pct = oi_acceleration(oi_vals)
    vol_accel, vol_ratio = volume_acceleration(df["volume"].values)
    taker_delta, taker_long, taker_short = taker_delta_info(taker_rows)

    price_chg = pct_change(df["close"].iloc[-1], df["close"].iloc[-2])
    rsi_now = float(df["rsi"].iloc[-1])
    smi_k = float(df["smi_k"].iloc[-1])
    smi_d = float(df["smi_d"].iloc[-1])

    tight = is_range_tight(df)
    up_wick = upper_wick_reject(df)
    low_wick = lower_wick_reject(df)
    up_break = breakout_up(df)
    down_break = breakout_down(df)

    long_early = 0
    long_confirm = 0
    short_early = 0
    short_confirm = 0

    if oi_accel:
        long_early += 15
    if vol_accel:
        long_early += 15
    if rsi_long_early(df):
        long_early += 15
    if smi_long_early(df):
        long_early += 15
    if tight:
        long_early += 10
    if PRICE_EARLY_LONG_MIN <= price_chg <= PRICE_EARLY_LONG_MAX:
        long_early += 10
    if oi_accel and vol_accel:
        long_early += 10
    if low_wick:
        long_early += 5

    if up_break:
        long_confirm += 20
    if taker_long:
        long_confirm += 20
    if oi_accel:
        long_confirm += 15
    if vol_accel:
        long_confirm += 15
    if df["rsi"].iloc[-1] > df["rsi"].iloc[-2]:
        long_confirm += 10
    if smi_k > smi_d:
        long_confirm += 10
    if 0.5 <= price_chg <= 2.8:
        long_confirm += 10

    if oi_accel:
        short_early += 15
    if vol_accel:
        short_early += 15
    if rsi_short_early(df):
        short_early += 15
    if smi_short_early(df):
        short_early += 15
    if tight:
        short_early += 10
    if PRICE_EARLY_SHORT_MIN <= price_chg <= PRICE_EARLY_SHORT_MAX:
        short_early += 10
    if oi_accel and vol_accel:
        short_early += 10
    if up_wick:
        short_early += 10

    if down_break:
        short_confirm += 20
    if taker_short:
        short_confirm += 20
    if oi_accel:
        short_confirm += 15
    if vol_accel:
        short_confirm += 15
    if df["rsi"].iloc[-1] < df["rsi"].iloc[-2]:
        short_confirm += 10
    if smi_k < smi_d:
        short_confirm += 10
    if -2.8 <= price_chg <= -0.5:
        short_confirm += 10

    return {
        "symbol": symbol,
        "price_chg": price_chg,
        "oi_pct": oi_pct,
        "vol_ratio": vol_ratio,
        "rsi": rsi_now,
        "smi_k": smi_k,
        "smi_d": smi_d,
        "taker_delta": taker_delta,
        "oi_accel": oi_accel,
        "vol_accel": vol_accel,
        "tight": tight,
        "up_wick": up_wick,
        "low_wick": low_wick,
        "up_break": up_break,
        "down_break": down_break,
        "taker_long": taker_long,
        "taker_short": taker_short,
        "long_early": long_early,
        "long_confirm": long_confirm,
        "short_early": short_early,
        "short_confirm": short_confirm,
    }


# =========================
# 알림 포맷
# =========================
def make_alert_text(kind, item):
    symbol = item["symbol"].replace("/USDT:USDT", "").replace("/USDT", "")
    price = item["price_chg"]
    oi_pct = item["oi_pct"]
    vol_ratio = item["vol_ratio"]
    taker_delta = item["taker_delta"]

    rsi = item["rsi"]
    smi_k = item["smi_k"]
    smi_d = item["smi_d"]

    long_early = item["long_early"]
    long_confirm = item["long_confirm"]
    short_early = item["short_early"]
    short_confirm = item["short_confirm"]

    tight_line = "range tight\n" if item["tight"] else ""
    low_wick_line = "lower wick reject\n" if item["low_wick"] else ""
    up_wick_line = "upper wick reject\n" if item["up_wick"] else ""
    up_break_line = "upper break yes\n" if item["up_break"] else ""
    down_break_line = "lower break yes\n" if item["down_break"] else ""

    oi_line = "(속도↑↑)" if item["oi_accel"] else ""
    oi_line_early = "(속도↑)" if item["oi_accel"] else ""
    vol_line = "(가속↑↑)" if item["vol_accel"] else ""
    vol_line_early = "(가속↑)" if item["vol_accel"] else ""

    if kind == "LONG_EARLY":
        text = (
            "🛰 <b>롱 예고</b>\n"
            + "⏰ " + now_str() + "\n\n"
            + "<b>" + symbol + "</b>\n"
            + "L-early " + str(long_early) + "\n"
            + "price {:+.2f}%\n".format(price)
            + "OI {:+.2f}% {}\n".format(oi_pct, oi_line_early)
            + "vol {:.2f}x {}\n".format(vol_ratio, vol_line_early)
            + "takerΔ {:+.2f}\n".format(taker_delta)
            + "RSI {:.1f} | SMI {:.1f}/{:.1f}\n".format(rsi, smi_k, smi_d)
            + tight_line
            + low_wick_line
        )
        return text.strip()

    if kind == "LONG_CONFIRM":
        text = (
            "🚀 <b>롱 발화</b>\n"
            + "⏰ " + now_str() + "\n\n"
            + "<b>" + symbol + "</b>\n"
            + "L-confirm " + str(long_confirm) + "\n"
            + "price {:+.2f}%\n".format(price)
            + "OI {:+.2f}% {}\n".format(oi_pct, oi_line)
            + "vol {:.2f}x {}\n".format(vol_ratio, vol_line)
            + "takerΔ {:+.2f}\n".format(taker_delta)
            + up_break_line
        )
        return text.strip()

    if kind == "SHORT_EARLY":
        text = (
            "🛰 <b>숏 예고</b>\n"
            + "⏰ " + now_str() + "\n\n"
            + "<b>" + symbol + "</b>\n"
            + "S-early " + str(short_early) + "\n"
            + "price {:+.2f}%\n".format(price)
            + "OI {:+.2f}% {}\n".format(oi_pct, oi_line_early)
            + "vol {:.2f}x {}\n".format(vol_ratio, vol_line_early)
            + "takerΔ {:+.2f}\n".format(taker_delta)
            + "RSI {:.1f} | SMI {:.1f}/{:.1f}\n".format(rsi, smi_k, smi_d)
            + tight_line
            + up_wick_line
        )
        return text.strip()

    text = (
        "💥 <b>숏 발화</b>\n"
        + "⏰ " + now_str() + "\n\n"
        + "<b>" + symbol + "</b>\n"
        + "S-confirm " + str(short_confirm) + "\n"
        + "price {:+.2f}%\n".format(price)
        + "OI {:+.2f}% {}\n".format(oi_pct, oi_line)
        + "vol {:.2f}x {}\n".format(vol_ratio, vol_line)
        + "takerΔ {:+.2f}\n".format(taker_delta)
        + down_break_line
    )
    return text.strip()


def should_send_alert(alert_key, cooldown_sec=900):
    now_ts = time.time()
    last_ts = last_alert_map.get(alert_key, 0)

    if now_ts - last_ts >= cooldown_sec:
        last_alert_map[alert_key] = now_ts
        return True
    return False


# =========================
# 실행
# =========================
def scan_once(symbols):
    scored = []

    total = len(symbols)
    for i, symbol in enumerate(symbols, start=1):
        try:
            print("[{}/{}] {}".format(i, total, symbol))
            item = score_symbol(symbol)
            if item is not None:
                scored.append(item)
            time.sleep(0.08)
        except Exception as e:
            print("[에러] {}: {}".format(symbol, e))

    if not scored:
        print("[스캔 결과 없음]")
        return

    best_long = sorted(
        scored,
        key=lambda x: (x["long_confirm"], x["long_early"]),
        reverse=True,
    )[:TOP_N_ALERTS]

    best_short = sorted(
        scored,
        key=lambda x: (x["short_confirm"], x["short_early"]),
        reverse=True,
    )[:TOP_N_ALERTS]

    print("\n[상위 롱 후보]")
    for item in best_long:
        print(
            "{} LC {} LE {} price {:.2f} oi {:.2f} vol {:.2f}".format(
                item["symbol"],
                item["long_confirm"],
                item["long_early"],
                item["price_chg"],
                item["oi_pct"],
                item["vol_ratio"],
            )
        )

    print("\n[상위 숏 후보]")
    for item in best_short:
        print(
            "{} SC {} SE {} price {:.2f} oi {:.2f} vol {:.2f}".format(
                item["symbol"],
                item["short_confirm"],
                item["short_early"],
                item["price_chg"],
                item["oi_pct"],
                item["vol_ratio"],
            )
        )

    for item in best_long:
        symbol = item["symbol"]

        if item["long_confirm"] >= LONG_CONFIRM_THRESHOLD:
            key = symbol + ":LONG_CONFIRM"
            if should_send_alert(key, cooldown_sec=900):
                send_telegram(make_alert_text("LONG_CONFIRM", item))

        elif item["long_early"] >= LONG_EARLY_THRESHOLD:
            key = symbol + ":LONG_EARLY"
            if should_send_alert(key, cooldown_sec=1200):
                send_telegram(make_alert_text("LONG_EARLY", item))

    for item in best_short:
        symbol = item["symbol"]

        if item["short_confirm"] >= SHORT_CONFIRM_THRESHOLD:
            key = symbol + ":SHORT_CONFIRM"
            if should_send_alert(key, cooldown_sec=900):
                send_telegram(make_alert_text("SHORT_CONFIRM", item))

        elif item["short_early"] >= SHORT_EARLY_THRESHOLD:
            key = symbol + ":SHORT_EARLY"
            if should_send_alert(key, cooldown_sec=1200):
                send_telegram(make_alert_text("SHORT_EARLY", item))


def main():
    symbols = get_usdt_perp_symbols()

    print("[시작] 심볼 수:", len(symbols))
    print("[알림] 텔레그램 설정됨:", bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID))

    start_msg = (
        "✅ <b>두식이아빠 코인 탐지기 v7 시작</b>\n"
        + "⏰ " + now_str() + "\n"
        + "timeframe: " + TIMEFRAME + "\n"
        + "symbols: " + str(len(symbols))
    )
    send_telegram(start_msg)

    print("\n[스캔 시작]", now_str())
    scan_once(symbols)
    print("[스캔 종료]", now_str())


if __name__ == "__main__":
    main()
