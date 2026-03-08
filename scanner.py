import os
import time
import json
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import ccxt
import numpy as np
import pandas as pd
import requests

# =========================================
# 기본 설정
# =========================================
TIMEFRAME = "3m"
HTF_1 = "15m"
HTF_2 = "1h"

OHLCV_LIMIT = 160
OI_PERIOD = "5m"
TAKER_PERIOD = "5m"
OI_LIMIT = 12
TAKER_LIMIT = 6

SCAN_INTERVAL_SEC = 60
SYMBOL_LIMIT = 60
TOP_N_ALERTS = 12

MAX_WORKERS = 8
HTF_WORKERS = 5

MIN_TURNOVER_20 = 700000
MIN_LAST_PRICE = 0.0005

LONG_EARLY_THRESHOLD = 58
SHORT_EARLY_THRESHOLD = 58
LONG_CONFIRM_THRESHOLD = 74
SHORT_CONFIRM_THRESHOLD = 74
SQUEEZE_THRESHOLD = 82

MA_FAST = 5
MA_MID = 10
MA_SLOW = 30
MA_SPREAD_5_10_MAX = 0.25  # %
MA_SPREAD_10_30_MAX = 0.40  # %

ALERT_STATE_FILE = "alert_state_v10_1.json"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# =========================================
# 전역 상태
# =========================================
last_alert_map = {}

exchange = ccxt.binance({
    "enableRateLimit": True,
    "options": {"defaultType": "future"},
})

_thread_local = threading.local()


# =========================================
# 세션 / HTTP
# =========================================
def get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update({"User-Agent": "scanner-v10.2"})
        _thread_local.session = s
    return _thread_local.session


def get_with_retry(url: str, params=None, timeout: int = 10, retries: int = 3):
    session = get_session()
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=timeout)

            if r.status_code == 429:
                wait = 1.0 * (attempt + 1)
                print(f"[429] wait {wait:.1f}s | {url}")
                time.sleep(wait)
                continue

            r.raise_for_status()
            return r
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(0.5 * (attempt + 1))

    raise RuntimeError("HTTP request failed")


# =========================================
# 유틸
# =========================================
def now_str() -> str:
    return datetime.now().strftime("%m-%d %H:%M:%S")


def safe_float(value, default=0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def pct_change(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a - b) / b * 100.0


def multi_bar_price_change(close: pd.Series, bars: int = 3) -> float:
    if len(close) < bars + 1:
        return 0.0
    return pct_change(float(close.iloc[-1]), float(close.iloc[-1 - bars]))


def binance_symbol(symbol: str) -> str:
    return symbol.replace("/", "").replace(":USDT", "")


def get_binance_futures_url(symbol: str) -> str:
    raw = symbol.replace("/USDT:USDT", "").replace("/USDT", "")
    return f"https://www.binance.com/en/futures/{raw}"


# =========================================
# 알림 상태 저장
# =========================================
def load_alert_state() -> None:
    global last_alert_map
    try:
        if os.path.exists(ALERT_STATE_FILE):
            with open(ALERT_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                last_alert_map = {k: float(v) for k, v in data.items()}
    except Exception as e:
        print(f"[알림 상태 로드 실패] {e}")


def save_alert_state() -> None:
    try:
        with open(ALERT_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(last_alert_map, f)
    except Exception as e:
        print(f"[알림 상태 저장 실패] {e}")


def should_send_alert(alert_key: str, cooldown_sec: int) -> bool:
    now_ts = time.time()
    last_ts = last_alert_map.get(alert_key, 0)
    if now_ts - last_ts >= cooldown_sec:
        last_alert_map[alert_key] = now_ts
        save_alert_state()
        return True
    return False


# =========================================
# 텔레그램
# =========================================
def send_telegram(text: str, silent: bool = False) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[텔레그램 미설정]")
        print(text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
        "disable_notification": silent,
    }

    try:
        session = get_session()
        resp = session.post(url, data=payload, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"[텔레그램 전송 실패] {e}")


# =========================================
# 바이낸스 데이터
# =========================================
def get_usdt_perp_symbols() -> list[str]:
    markets = exchange.load_markets()
    symbols = []
    for sym, market in markets.items():
        if not market.get("active", True):
            continue
        if market.get("quote") != "USDT":
            continue
        if not market.get("swap", False):
            continue
        symbols.append(sym)
    return sorted(symbols)


def fetch_ohlcv(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().reset_index(drop=True)


def fetch_24h_ticker_map() -> dict:
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    try:
        r = get_with_retry(url, timeout=10)
        rows = r.json()
        out = {}
        for row in rows:
            sym = row.get("symbol", "")
            out[sym] = {
                "quoteVolume": safe_float(row.get("quoteVolume")),
                "lastPrice": safe_float(row.get("lastPrice")),
            }
        return out
    except Exception as e:
        print(f"[24h ticker 조회 실패] {e}")
        return {}


def fetch_open_interest_history(symbol: str, period: str = OI_PERIOD, limit: int = OI_LIMIT) -> list[float]:
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    params = {
        "symbol": binance_symbol(symbol),
        "period": period,
        "limit": limit,
    }
    try:
        r = get_with_retry(url, params=params, timeout=10)
        rows = r.json()
        return [safe_float(x.get("sumOpenInterest")) for x in rows]
    except Exception:
        return []


def fetch_taker_ratio(symbol: str, period: str = TAKER_PERIOD, limit: int = TAKER_LIMIT) -> list[dict]:
    url = "https://fapi.binance.com/futures/data/takerlongshortRatio"
    params = {
        "symbol": binance_symbol(symbol),
        "period": period,
        "limit": limit,
    }
    try:
        r = get_with_retry(url, params=params, timeout=10)
        rows = r.json()
        out = []
        for row in rows:
            out.append({
                "buySellRatio": safe_float(row.get("buySellRatio"), 1.0),
                "buyVol": safe_float(row.get("buyVol")),
                "sellVol": safe_float(row.get("sellVol")),
            })
        return out
    except Exception:
        return []


def fetch_funding_rate(symbol: str) -> float:
    url = "https://fapi.binance.com/fapi/v1/premiumIndex"
    params = {"symbol": binance_symbol(symbol)}
    try:
        r = get_with_retry(url, params=params, timeout=10)
        return safe_float(r.json().get("lastFundingRate"))
    except Exception:
        return 0.0


# =========================================
# 지표
# =========================================
def calc_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calc_smi(df: pd.DataFrame, length_k: int = 10, length_d: int = 3, ema_len: int = 3) -> pd.DataFrame:
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


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = calc_rsi(out["close"], 14)
    out["ema7"] = out["close"].ewm(span=7, adjust=False).mean()
    out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["ema50"] = out["close"].ewm(span=50, adjust=False).mean()
    out["ma5"] = out["close"].rolling(MA_FAST).mean()
    out["ma10"] = out["close"].rolling(MA_MID).mean()
    out["ma30"] = out["close"].rolling(MA_SLOW).mean()
    out["range"] = out["high"] - out["low"]
    out["body"] = (out["close"] - out["open"]).abs()
    out["turnover"] = out["close"] * out["volume"]
    out["vol_ma20"] = out["volume"].rolling(20).mean()
    out["turnover_ma20"] = out["turnover"].rolling(20).mean()
    out = calc_smi(out)
    return out.dropna().reset_index(drop=True)


# =========================================
# 조건 / 가속도
# =========================================
def calc_acceleration_from_list(values: list[float], window: int = 6) -> float:
    if len(values) < max(window + 2, 8):
        return 0.0

    s = pd.Series(values, dtype="float64")
    velocity = s.diff()
    accel = velocity.diff()
    avg_velocity = velocity.abs().rolling(window).mean()

    latest_accel = safe_float(accel.iloc[-1], 0.0)
    base = safe_float(avg_velocity.iloc[-1], 0.0)

    if base == 0:
        return 0.0
    return latest_accel / base


def volume_acceleration(vols: np.ndarray, window: int = 10) -> tuple[float, float]:
    if len(vols) < max(window + 2, 20):
        return 0.0, 0.0

    s = pd.Series(vols, dtype="float64")
    velocity = s.diff()
    accel = velocity.diff()
    avg_velocity = velocity.abs().rolling(window).mean()

    accel_score = safe_float(accel.iloc[-1], 0.0) / max(safe_float(avg_velocity.iloc[-1], 0.0), 1e-9)
    vol_ratio = safe_float(s.iloc[-1], 0.0) / max(safe_float(s.tail(20).mean(), 0.0), 1e-9)
    return accel_score, vol_ratio


def oi_state(oi_vals: list[float]) -> dict:
    if len(oi_vals) < 4:
        return {
            "accel_up": False,
            "accel_down": False,
            "latest_pct": 0.0,
            "accel_score": 0.0,
        }

    d1 = oi_vals[-1] - oi_vals[-2]
    d2 = oi_vals[-2] - oi_vals[-3]
    d3 = oi_vals[-3] - oi_vals[-4]

    accel_up = d1 > d2 > d3 and d1 > 0
    accel_down = d1 < d2 < d3 and d1 < 0
    latest_pct = pct_change(oi_vals[-1], oi_vals[-2])
    accel_score = calc_acceleration_from_list(oi_vals, window=6)

    return {
        "accel_up": accel_up,
        "accel_down": accel_down,
        "latest_pct": latest_pct,
        "accel_score": accel_score,
    }


def taker_delta_info(taker_rows: list[dict]) -> tuple[float, bool, bool, float]:
    if len(taker_rows) < 2:
        return 0.0, False, False, 1.0

    latest = taker_rows[-1]
    prev = taker_rows[-2]

    latest_delta = latest["buyVol"] - latest["sellVol"]
    prev_delta = prev["buyVol"] - prev["sellVol"]

    long_bias = latest_delta > 0 and latest_delta > prev_delta
    short_bias = latest_delta < 0 and latest_delta < prev_delta
    buy_sell_ratio = safe_float(latest.get("buySellRatio"), 1.0)

    return latest_delta, long_bias, short_bias, buy_sell_ratio


def taker_imbalance_score(taker_ratio: float, vol_ratio: float) -> tuple[int, int]:
    long_score = 0
    short_score = 0

    if taker_ratio >= 1.25:
        long_score += 6
    if taker_ratio >= 1.50 and vol_ratio > 1.2:
        long_score += 8
    if taker_ratio >= 1.80 and vol_ratio > 1.5:
        long_score += 8

    if taker_ratio <= 0.80:
        short_score += 6
    if taker_ratio <= 0.67 and vol_ratio > 1.2:
        short_score += 8
    if taker_ratio <= 0.55 and vol_ratio > 1.5:
        short_score += 8

    return long_score, short_score


def is_range_tight(df: pd.DataFrame) -> bool:
    recent = df["range"].tail(5).mean()
    base = df["range"].tail(30).mean()
    if pd.isna(recent) or pd.isna(base) or base == 0:
        return False
    return recent < base * 0.72


def upper_wick_reject(df: pd.DataFrame) -> bool:
    row = df.iloc[-1]
    upper = row["high"] - max(row["open"], row["close"])
    body = abs(row["close"] - row["open"])
    return upper > body * 1.2 and row["close"] < row["high"]


def lower_wick_reject(df: pd.DataFrame) -> bool:
    row = df.iloc[-1]
    lower = min(row["open"], row["close"]) - row["low"]
    body = abs(row["close"] - row["open"])
    return lower > body * 1.2 and row["close"] > row["low"]


def breakout_up(df: pd.DataFrame) -> bool:
    prev_high = df["high"].iloc[-6:-1].max()
    return df["close"].iloc[-1] > prev_high


def breakout_down(df: pd.DataFrame) -> bool:
    prev_low = df["low"].iloc[-6:-1].min()
    return df["close"].iloc[-1] < prev_low


def rsi_long_early(df: pd.DataFrame) -> bool:
    r1, r2, r3 = df["rsi"].iloc[-1], df["rsi"].iloc[-2], df["rsi"].iloc[-3]
    return (r1 > r2 > r3) and (r1 < 56)


def rsi_short_early(df: pd.DataFrame) -> bool:
    r1, r2, r3 = df["rsi"].iloc[-1], df["rsi"].iloc[-2], df["rsi"].iloc[-3]
    return (r1 < r2 < r3) and (r1 > 44)


def smi_bull_cross(df: pd.DataFrame) -> bool:
    k1, k2 = df["smi_k"].iloc[-1], df["smi_k"].iloc[-2]
    d1, d2 = df["smi_d"].iloc[-1], df["smi_d"].iloc[-2]
    return k2 <= d2 and k1 > d1


def smi_bear_cross(df: pd.DataFrame) -> bool:
    k1, k2 = df["smi_k"].iloc[-1], df["smi_k"].iloc[-2]
    d1, d2 = df["smi_d"].iloc[-1], df["smi_d"].iloc[-2]
    return k2 >= d2 and k1 < d1


def smi_bull_cross_strong(df: pd.DataFrame) -> bool:
    k1, k2 = df["smi_k"].iloc[-1], df["smi_k"].iloc[-2]
    d1, d2 = df["smi_d"].iloc[-1], df["smi_d"].iloc[-2]
    return k2 <= d2 and k1 > d1 and min(k1, d1) < -40


def smi_bear_cross_strong(df: pd.DataFrame) -> bool:
    k1, k2 = df["smi_k"].iloc[-1], df["smi_k"].iloc[-2]
    d1, d2 = df["smi_d"].iloc[-1], df["smi_d"].iloc[-2]
    return k2 >= d2 and k1 < d1 and max(k1, d1) > 40


def trend_filter(df: pd.DataFrame) -> dict:
    row = df.iloc[-1]
    long_ok = row["ema7"] > row["ema20"] and row["close"] > row["ema20"]
    short_ok = row["ema7"] < row["ema20"] and row["close"] < row["ema20"]
    return {"long_ok": long_ok, "short_ok": short_ok}


def ma_cluster_info(df: pd.DataFrame) -> dict:
    row = df.iloc[-1]
    close_now = safe_float(row["close"], 0.0)
    ma5 = safe_float(row.get("ma5"), close_now)
    ma10 = safe_float(row.get("ma10"), close_now)
    ma30 = safe_float(row.get("ma30"), close_now)

    if close_now <= 0:
        return {
            "squeeze": False,
            "spread_5_10": 999.0,
            "spread_10_30": 999.0,
            "long_bias": False,
            "short_bias": False,
            "cross_up": False,
            "cross_down": False,
        }

    spread_5_10 = abs(ma5 - ma10) / close_now * 100.0
    spread_10_30 = abs(ma10 - ma30) / close_now * 100.0
    squeeze = spread_5_10 <= MA_SPREAD_5_10_MAX and spread_10_30 <= MA_SPREAD_10_30_MAX

    prev = df.iloc[-2] if len(df) >= 2 else row
    prev_close = safe_float(prev["close"], close_now)
    prev_ma5 = safe_float(prev.get("ma5"), prev_close)
    prev_ma10 = safe_float(prev.get("ma10"), prev_close)
    prev_ma30 = safe_float(prev.get("ma30"), prev_close)

    long_bias = close_now > ma5 > ma10 > ma30 or (close_now > ma5 > ma10 and ma10 >= ma30)
    short_bias = close_now < ma5 < ma10 < ma30 or (close_now < ma5 < ma10 and ma10 <= ma30)

    cross_up = prev_close <= prev_ma5 and close_now > ma5 and ma5 >= ma10
    cross_down = prev_close >= prev_ma5 and close_now < ma5 and ma5 <= ma10

    return {
        "squeeze": squeeze,
        "spread_5_10": spread_5_10,
        "spread_10_30": spread_10_30,
        "long_bias": long_bias,
        "short_bias": short_bias,
        "cross_up": cross_up,
        "cross_down": cross_down,
    }


def is_tradeable(df: pd.DataFrame) -> bool:
    turnover_usdt = float(df["turnover"].tail(20).mean())
    close_now = float(df["close"].iloc[-1])
    return close_now > MIN_LAST_PRICE and turnover_usdt > MIN_TURNOVER_20


def detect_short_squeeze(price_chg_3: float, oi_pct: float) -> bool:
    return price_chg_3 > 0.6 and oi_pct < -0.2


def detect_long_liq(price_chg_3: float, oi_pct: float) -> bool:
    return price_chg_3 < -0.6 and oi_pct < -0.2


def relative_strength(symbol_change: float, btc_change: float) -> float:
    return symbol_change - btc_change


def long_confirm_gate(item: dict) -> bool:
    conds = 0
    if item["up_break"]:
        conds += 1
    if item["taker_long"]:
        conds += 1
    if item["price_chg_3"] >= 0.7:
        conds += 1
    if item["vol_ratio"] > 1.5:
        conds += 1
    return conds >= 2


def short_confirm_gate(item: dict) -> bool:
    conds = 0
    if item["down_break"]:
        conds += 1
    if item["taker_short"]:
        conds += 1
    if item["price_chg_3"] <= -0.7:
        conds += 1
    if item["vol_ratio"] > 1.5:
        conds += 1
    return conds >= 2


# =========================================
# BTC / HTF
# =========================================
def get_btc_bias() -> dict:
    try:
        btc = add_indicators(fetch_ohlcv("BTC/USDT:USDT", "15m", 140))
        row = btc.iloc[-1]
        btc_price3 = multi_bar_price_change(btc["close"], 3)

        strong_up = row["ema7"] > row["ema20"] and btc_price3 > 0.40
        strong_down = row["ema7"] < row["ema20"] and btc_price3 < -0.40

        label = "up" if strong_up else "down" if strong_down else "neutral"
        return {
            "label": label,
            "strong_up": strong_up,
            "strong_down": strong_down,
            "price_chg_3": btc_price3,
        }
    except Exception:
        return {
            "label": "neutral",
            "strong_up": False,
            "strong_down": False,
            "price_chg_3": 0.0,
        }


def get_single_htf(sym: str) -> tuple[str, dict]:
    try:
        df15 = add_indicators(fetch_ohlcv(sym, HTF_1, 120))
        df1h = add_indicators(fetch_ohlcv(sym, HTF_2, 120))
        tf15 = trend_filter(df15)
        tf1h = trend_filter(df1h)

        return sym, {
            "long_bias": tf15["long_ok"] or tf1h["long_ok"],
            "short_bias": tf15["short_ok"] or tf1h["short_ok"],
        }
    except Exception:
        return sym, {"long_bias": False, "short_bias": False}


def build_htf_cache_parallel(symbols: list[str]) -> dict:
    cache = {}
    with ThreadPoolExecutor(max_workers=min(HTF_WORKERS, MAX_WORKERS)) as executor:
        futures = {executor.submit(get_single_htf, sym): sym for sym in symbols}
        for future in as_completed(futures):
            try:
                sym, result = future.result()
                cache[sym] = result
            except Exception:
                pass
    return cache


def get_market_score(htf_cache: dict, symbols: list[str]) -> float:
    valid = [s for s in symbols if s in htf_cache]
    if not valid:
        return 50.0
    long_count = sum(1 for s in valid if htf_cache[s]["long_bias"])
    return (long_count / len(valid)) * 100.0


# =========================================
# 점수 계산
# =========================================
def score_symbol(symbol: str, btc_bias: dict, htf_cache: dict, market_score: float) -> dict | None:
    df = fetch_ohlcv(symbol, TIMEFRAME, OHLCV_LIMIT)
    if len(df) < 80:
        return None

    df = add_indicators(df)
    if len(df) < 60 or not is_tradeable(df):
        return None

    oi_vals = fetch_open_interest_history(symbol, period=OI_PERIOD, limit=OI_LIMIT)
    taker_rows = fetch_taker_ratio(symbol, period=TAKER_PERIOD, limit=TAKER_LIMIT)
    funding = fetch_funding_rate(symbol)

    oi_info = oi_state(oi_vals)
    oi_accel_up = oi_info["accel_up"]
    oi_accel_down = oi_info["accel_down"]
    oi_pct = oi_info["latest_pct"]
    oi_accel_score = oi_info["accel_score"]

    vol_accel_score, vol_ratio = volume_acceleration(df["volume"].values)
    taker_delta, taker_long, taker_short, taker_ratio = taker_delta_info(taker_rows)
    taker_long_score, taker_short_score = taker_imbalance_score(taker_ratio, vol_ratio)

    price_chg_1 = pct_change(df["close"].iloc[-1], df["close"].iloc[-2])
    price_chg_3 = multi_bar_price_change(df["close"], 3)

    btc_price3 = btc_bias.get("price_chg_3", 0.0)
    rs_score = relative_strength(price_chg_3, btc_price3)

    rsi_now = float(df["rsi"].iloc[-1])
    smi_k = float(df["smi_k"].iloc[-1])
    smi_d = float(df["smi_d"].iloc[-1])
    current_price = float(df["close"].iloc[-1])

    tight = is_range_tight(df)
    up_wick = upper_wick_reject(df)
    low_wick = lower_wick_reject(df)
    up_break = breakout_up(df)
    down_break = breakout_down(df)

    tf_now = trend_filter(df)
    ma_info = ma_cluster_info(df)
    htf_bias = htf_cache.get(symbol, {"long_bias": False, "short_bias": False})

    short_squeeze_flag = detect_short_squeeze(price_chg_3, oi_pct)
    long_liq_flag = detect_long_liq(price_chg_3, oi_pct)

    long_early = 0
    long_confirm = 0
    long_squeeze = 0

    short_early = 0
    short_confirm = 0
    short_squeeze = 0

    # LONG EARLY
    if rsi_long_early(df):
        long_early += 10
    if smi_bull_cross(df):
        long_early += 12
    if tight:
        long_early += 8
    if ma_info["squeeze"]:
        long_early += 10
    if ma_info["cross_up"]:
        long_early += 10
    if ma_info["long_bias"]:
        long_early += 8
    if 0.2 <= price_chg_3 <= 1.8:
        long_early += 10
    if low_wick:
        long_early += 6
    if tf_now["long_ok"]:
        long_early += 8
    if htf_bias["long_bias"]:
        long_early += 8
    if oi_accel_up:
        long_early += 8
    if oi_accel_score > 1.2:
        long_early += 8
    if vol_accel_score > 1.2:
        long_early += 8
    if vol_ratio > 1.2:
        long_early += 6
    long_early += taker_long_score
    if rs_score > 0.8:
        long_early += 8
    if rs_score > 1.3:
        long_early += 6
    if funding < -0.001:
        long_early += 6
    if funding < -0.002:
        long_early += 6
    if short_squeeze_flag:
        long_early += 10

    # LONG CONFIRM
    if up_break:
        long_confirm += 18
    if ma_info["squeeze"]:
        long_confirm += 8
    if ma_info["cross_up"]:
        long_confirm += 8
    if ma_info["long_bias"]:
        long_confirm += 8
    if taker_long:
        long_confirm += 14
    if taker_ratio >= 1.4:
        long_confirm += 8
    if 0.6 <= price_chg_3 <= 3.0:
        long_confirm += 10
    if vol_ratio > 1.5:
        long_confirm += 10
    if vol_accel_score > 1.4:
        long_confirm += 8
    if oi_accel_up:
        long_confirm += 10
    if oi_accel_score > 1.4:
        long_confirm += 8
    if df["rsi"].iloc[-1] > df["rsi"].iloc[-2]:
        long_confirm += 6
    if smi_bull_cross(df):
        long_confirm += 8
    if smi_bull_cross_strong(df):
        long_confirm += 8
    if tf_now["long_ok"]:
        long_confirm += 8
    if htf_bias["long_bias"]:
        long_confirm += 6
    if rs_score > 1.0:
        long_confirm += 8
    if rs_score > 1.8:
        long_confirm += 6
    if funding < -0.001:
        long_confirm += 6
    if market_score >= 60:
        long_confirm += 6

    # LONG SQUEEZE
    if short_squeeze_flag:
        long_squeeze += 24
    if ma_info["squeeze"] and ma_info["cross_up"]:
        long_squeeze += 10
    if price_chg_3 > 0.8 and oi_pct < -0.2:
        long_squeeze += 12
    if price_chg_3 > 1.2 and oi_pct < -0.4:
        long_squeeze += 12
    if taker_ratio >= 1.5:
        long_squeeze += 10
    if vol_ratio > 1.5:
        long_squeeze += 10
    if rs_score > 1.2:
        long_squeeze += 8
    if funding < -0.0015:
        long_squeeze += 8

    # SHORT EARLY
    if rsi_short_early(df):
        short_early += 10
    if smi_bear_cross(df):
        short_early += 12
    if tight:
        short_early += 8
    if ma_info["squeeze"]:
        short_early += 10
    if ma_info["cross_down"]:
        short_early += 10
    if ma_info["short_bias"]:
        short_early += 8
    if -1.8 <= price_chg_3 <= -0.2:
        short_early += 10
    if up_wick:
        short_early += 6
    if tf_now["short_ok"]:
        short_early += 8
    if htf_bias["short_bias"]:
        short_early += 8
    if oi_accel_up and price_chg_3 < -0.4:
        short_early += 8
    if oi_accel_down and price_chg_3 < -0.4:
        short_early += 6
    if oi_accel_score < -1.2:
        short_early += 8
    if vol_accel_score > 1.2:
        short_early += 8
    if vol_ratio > 1.2:
        short_early += 6
    short_early += taker_short_score
    if rs_score < -0.8:
        short_early += 8
    if rs_score < -1.3:
        short_early += 6
    if funding > 0.001:
        short_early += 6
    if funding > 0.002:
        short_early += 6
    if long_liq_flag:
        short_early += 10

    # SHORT CONFIRM
    if down_break:
        short_confirm += 18
    if ma_info["squeeze"]:
        short_confirm += 8
    if ma_info["cross_down"]:
        short_confirm += 8
    if ma_info["short_bias"]:
        short_confirm += 8
    if taker_short:
        short_confirm += 14
    if taker_ratio <= 0.72:
        short_confirm += 8
    if -3.0 <= price_chg_3 <= -0.6:
        short_confirm += 10
    if vol_ratio > 1.5:
        short_confirm += 10
    if vol_accel_score > 1.4:
        short_confirm += 8
    if price_chg_3 < -0.5 and oi_accel_up:
        short_confirm += 10
    if oi_accel_score < -1.4:
        short_confirm += 8
    if df["rsi"].iloc[-1] < df["rsi"].iloc[-2]:
        short_confirm += 6
    if smi_bear_cross(df):
        short_confirm += 8
    if smi_bear_cross_strong(df):
        short_confirm += 8
    if tf_now["short_ok"]:
        short_confirm += 8
    if htf_bias["short_bias"]:
        short_confirm += 6
    if rs_score < -1.0:
        short_confirm += 8
    if rs_score < -1.8:
        short_confirm += 6
    if funding > 0.001:
        short_confirm += 6
    if market_score <= 40:
        short_confirm += 6

    # SHORT SQUEEZE / LONG LIQ
    if long_liq_flag:
        short_squeeze += 24
    if ma_info["squeeze"] and ma_info["cross_down"]:
        short_squeeze += 10
    if price_chg_3 < -0.8 and oi_pct < -0.2:
        short_squeeze += 12
    if price_chg_3 < -1.2 and oi_pct < -0.4:
        short_squeeze += 12
    if taker_ratio <= 0.70:
        short_squeeze += 10
    if vol_ratio > 1.5:
        short_squeeze += 10
    if rs_score < -1.2:
        short_squeeze += 8
    if funding > 0.0015:
        short_squeeze += 8

    # BTC 필터
    if btc_bias["strong_down"]:
        long_early -= 8
        long_confirm -= 10
        long_squeeze -= 8

    if btc_bias["strong_up"]:
        short_early -= 8
        short_confirm -= 10
        short_squeeze -= 8

    if btc_bias["strong_down"] and rs_score > 1.3:
        long_early += 8
        long_confirm += 8
    if btc_bias["strong_up"] and rs_score < -1.3:
        short_early += 8
        short_confirm += 8

    # 동시신호 방지
    if abs(long_confirm - short_confirm) < 8 and abs(long_early - short_early) < 8 and abs(long_squeeze - short_squeeze) < 8:
        return None

    return {
        "symbol": symbol,
        "current_price": current_price,
        "price_chg": price_chg_1,
        "price_chg_3": price_chg_3,
        "btc_price_chg_3": btc_price3,
        "rs_score": rs_score,
        "oi_pct": oi_pct,
        "oi_accel_score": oi_accel_score,
        "vol_accel_score": vol_accel_score,
        "vol_ratio": vol_ratio,
        "funding": funding,
        "rsi": rsi_now,
        "smi_k": smi_k,
        "smi_d": smi_d,
        "taker_delta": taker_delta,
        "taker_ratio": taker_ratio,
        "taker_long": taker_long,
        "taker_short": taker_short,
        "oi_accel_up": oi_accel_up,
        "oi_accel_down": oi_accel_down,
        "up_break": up_break,
        "down_break": down_break,
        "htf_long": htf_bias["long_bias"],
        "htf_short": htf_bias["short_bias"],
        "ma_squeeze": ma_info["squeeze"],
        "ma_cross_up": ma_info["cross_up"],
        "ma_cross_down": ma_info["cross_down"],
        "ma_long_bias": ma_info["long_bias"],
        "ma_short_bias": ma_info["short_bias"],
        "ma_spread_5_10": ma_info["spread_5_10"],
        "ma_spread_10_30": ma_info["spread_10_30"],
        "btc_label": btc_bias["label"],
        "market_score": market_score,
        "short_squeeze_detected": short_squeeze_flag,
        "long_liq_detected": long_liq_flag,
        "long_early": long_early,
        "long_confirm": long_confirm,
        "long_squeeze": long_squeeze,
        "short_early": short_early,
        "short_confirm": short_confirm,
        "short_squeeze": short_squeeze,
    }


# =========================================
# 알림 포맷
# =========================================
def make_alert_text(kind: str, item: dict) -> str:
    symbol = item["symbol"].replace("/USDT:USDT", "").replace("/USDT", "")
    funding_pct = item["funding"] * 100

    header_map = {
        "LONG_EARLY": "🛰 <b>[LONG EARLY]</b>",
        "LONG_CONFIRM": "🚀 <b>[LONG CONFIRM]</b>",
        "LONG_SQUEEZE": "⚡ <b>[SHORT SQUEEZE]</b>",
        "SHORT_EARLY": "🛰 <b>[SHORT EARLY]</b>",
        "SHORT_CONFIRM": "💥 <b>[SHORT CONFIRM]</b>",
        "SHORT_SQUEEZE": "⚠️ <b>[LONG LIQUIDATION]</b>",
    }

    score_key_map = {
        "LONG_EARLY": "long_early",
        "LONG_CONFIRM": "long_confirm",
        "LONG_SQUEEZE": "long_squeeze",
        "SHORT_EARLY": "short_early",
        "SHORT_CONFIRM": "short_confirm",
        "SHORT_SQUEEZE": "short_squeeze",
    }

    color_map = {
        "LONG_EARLY": "green",
        "LONG_CONFIRM": "green",
        "LONG_SQUEEZE": "green",
        "SHORT_EARLY": "red",
        "SHORT_CONFIRM": "red",
        "SHORT_SQUEEZE": "red",
    }

    header = header_map.get(kind, "🔔 <b>[ALERT]</b>")
    score_key = score_key_map.get(kind, "")
    score_val = float(item.get(score_key, 0.0))

    bar_count = max(0, min(int(score_val / 10), 10))
    if color_map.get(kind) == "green":
        status_bar = "🟢" * bar_count + "⚪" * (10 - bar_count)
    else:
        status_bar = "🔴" * bar_count + "⚪" * (10 - bar_count)

    comment = ""
    if kind == "LONG_SQUEEZE":
        comment = "price up + OI down = short liquidation"
    elif kind == "SHORT_SQUEEZE":
        comment = "price down + OI down = long liquidation"
    elif kind == "LONG_CONFIRM" and item.get("up_break"):
        comment = "breakout + flow aligned"
    elif kind == "SHORT_CONFIRM" and item.get("down_break"):
        comment = "breakdown + flow aligned"

    lines = [
        header,
        f"<b>{symbol}</b> | Price <b>{item['current_price']:.6f}</b>",
        f"Score <b>{score_val:.1f}</b> | {status_bar}",
        "",
        "📊 <b>Momentum</b>",
        f"├ 3-Bar Chg: <code>{item['price_chg_3']:+.2f}%</code>",
        f"├ RS Score: <code>{item['rs_score']:+.2f}</code>",
        f"└ RSI/SMI: <code>{item['rsi']:.1f}</code> / <code>{item['smi_k']:.1f},{item['smi_d']:.1f}</code>",
        "",
        "📐 <b>MA 5/10/30</b>",
        f"├ Squeeze: <b>{'YES' if item['ma_squeeze'] else 'NO'}</b>",
        f"├ 5-10 / 10-30: <code>{item['ma_spread_5_10']:.3f}%</code> / <code>{item['ma_spread_10_30']:.3f}%</code>",
        f"└ Bias/Cross: <code>{'L' if item['ma_long_bias'] else 'S' if item['ma_short_bias'] else '-'}</code> / <code>{'UP' if item['ma_cross_up'] else 'DOWN' if item['ma_cross_down'] else '-'}</code>",
        "",
        "🔌 <b>Liquidity & Flow</b>",
        f"├ OI: <code>{item['oi_pct']:+.2f}%</code> | Accel <code>{item['oi_accel_score']:+.2f}</code>",
        f"├ Vol: <code>{item['vol_ratio']:.2f}x</code> | Accel <code>{item['vol_accel_score']:+.2f}</code>",
        f"└ Taker: <code>{item['taker_ratio']:.2f}</code> | Δ <code>{item['taker_delta']:+.2f}</code>",
        "",
        "🌐 <b>Market Env</b>",
        f"├ BTC: <b>{item['btc_label'].upper()}</b> <code>{item['btc_price_chg_3']:+.2f}%</code>",
        f"├ Funding: <code>{funding_pct:+.4f}%</code>",
        f"└ Breadth: <code>{item['market_score']:.1f}</code>",
    ]

    if comment:
        lines.extend(["", f"📝 <i>{comment}</i>"])

    chart_url = get_binance_futures_url(item["symbol"])
    lines.extend([
        "",
        f"🔗 <a href=\"{chart_url}\">바이낸스 차트 열기</a>",
        f"⏰ {now_str()}",
    ])
    return "\n".join(lines)


# =========================================
# 스캔 루프
# =========================================
def pick_top_symbols(symbols: list[str]) -> list[str]:
    ticker_map = fetch_24h_ticker_map()
    ranked = []

    for sym in symbols:
        bsym = binance_symbol(sym)
        info = ticker_map.get(bsym, {})
        qv = safe_float(info.get("quoteVolume"), 0.0)
        lp = safe_float(info.get("lastPrice"), 0.0)
        if qv <= 0 or lp <= 0:
            continue
        ranked.append((sym, qv))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return [sym for sym, _ in ranked[:SYMBOL_LIMIT]]


def build_item_safe(symbol: str, btc_bias: dict, htf_cache: dict, market_score: float) -> dict | None:
    try:
        return score_symbol(symbol, btc_bias, htf_cache, market_score)
    except Exception as e:
        print(f"[에러] {symbol}: {e}")
        return None


def print_top_debug(scored: list[dict]) -> None:
    print("\n[상위 후보 디버그]")
    top = sorted(
        scored,
        key=lambda z: max(
            z["long_early"], z["long_confirm"], z["long_squeeze"],
            z["short_early"], z["short_confirm"], z["short_squeeze"]
        ),
        reverse=True
    )[:12]

    for x in top:
        print(
            x["symbol"],
            "| L", round(x["long_early"], 1), round(x["long_confirm"], 1), round(x["long_squeeze"], 1),
            "| S", round(x["short_early"], 1), round(x["short_confirm"], 1), round(x["short_squeeze"], 1),
            "| p3", round(x["price_chg_3"], 2),
            "| rs", round(x["rs_score"], 2),
            "| oi", round(x["oi_pct"], 2),
            "| vol", round(x["vol_ratio"], 2),
            "| btc", x["btc_label"],
        )


def scan_once(symbols: list[str]) -> None:
    scored = []
    btc_bias = get_btc_bias()

    print(f"[BTC bias] {btc_bias['label']} (chg3 {btc_bias['price_chg_3']:+.2f}%)")
    print("[HTF 캐시 생성 중...]")
    htf_cache = build_htf_cache_parallel(symbols)
    market_score = get_market_score(htf_cache, symbols)
    print(f"[Market Score] {market_score:.1f}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(build_item_safe, sym, btc_bias, htf_cache, market_score): sym
            for sym in symbols
        }
        for future in as_completed(futures):
            item = future.result()
            if item:
                scored.append(item)

    if not scored:
        print("[스캔 결과 없음]")
        return

    print_top_debug(scored)

    best_long = sorted(
        scored,
        key=lambda x: (x["long_squeeze"], x["long_confirm"], x["long_early"]),
        reverse=True
    )[:TOP_N_ALERTS]

    best_short = sorted(
        scored,
        key=lambda x: (x["short_squeeze"], x["short_confirm"], x["short_early"]),
        reverse=True
    )[:TOP_N_ALERTS]

    # LONG 계열
    for item in best_long:
        symbol = item["symbol"]

        if item["long_squeeze"] >= SQUEEZE_THRESHOLD:
            key = f"{symbol}:LONG_SQUEEZE"
            if should_send_alert(key, cooldown_sec=900):
                send_telegram(make_alert_text("LONG_SQUEEZE", item), silent=False)

        elif item["long_confirm"] >= LONG_CONFIRM_THRESHOLD and long_confirm_gate(item):
            key = f"{symbol}:LONG_CONFIRM"
            if should_send_alert(key, cooldown_sec=900):
                send_telegram(make_alert_text("LONG_CONFIRM", item), silent=False)

        elif item["long_early"] >= LONG_EARLY_THRESHOLD:
            key = f"{symbol}:LONG_EARLY"
            if should_send_alert(key, cooldown_sec=1200):
                send_telegram(make_alert_text("LONG_EARLY", item), silent=True)

    # SHORT 계열
    for item in best_short:
        symbol = item["symbol"]

        if item["short_squeeze"] >= SQUEEZE_THRESHOLD:
            key = f"{symbol}:SHORT_SQUEEZE"
            if should_send_alert(key, cooldown_sec=900):
                send_telegram(make_alert_text("SHORT_SQUEEZE", item), silent=False)

        elif item["short_confirm"] >= SHORT_CONFIRM_THRESHOLD and short_confirm_gate(item):
            key = f"{symbol}:SHORT_CONFIRM"
            if should_send_alert(key, cooldown_sec=900):
                send_telegram(make_alert_text("SHORT_CONFIRM", item), silent=False)

        elif item["short_early"] >= SHORT_EARLY_THRESHOLD:
            key = f"{symbol}:SHORT_EARLY"
            if should_send_alert(key, cooldown_sec=1200):
                send_telegram(make_alert_text("SHORT_EARLY", item), silent=True)


def main() -> None:
    load_alert_state()

    all_symbols = get_usdt_perp_symbols()
    symbols = pick_top_symbols(all_symbols)

    print(f"[시작] 전체 심볼 수: {len(all_symbols)}")
    print(f"[시작] 스캔 대상 수: {len(symbols)}")
    print(f"[설정] workers={MAX_WORKERS}, htf_workers={HTF_WORKERS}, interval={SCAN_INTERVAL_SEC}s")

    send_telegram(
        f"✅ <b>두식이아빠 코인 탐지기 v10.2 시작</b>\n"
        f"⏰ {now_str()}\n"
        f"timeframe: {TIMEFRAME}\n"
        f"htf: {HTF_1}, {HTF_2}\n"
        f"symbols: {len(symbols)} / {len(all_symbols)}\n"
        f"workers: {MAX_WORKERS}\n"
        f"mode: BTC filter + RS + squeeze + acceleration + UI patch + MA 5/10/30 squeeze",
        silent=False
    )

    while True:
        started = time.time()
        print(f"\n[스캔 시작] {now_str()}")

        try:
            scan_once(symbols)
        except Exception as e:
            print(f"[치명 오류] {e}")

        elapsed = time.time() - started
        sleep_sec = max(5, SCAN_INTERVAL_SEC - int(elapsed))
        print(f"[스캔 종료] 소요 {elapsed:.1f}s / 다음 스캔 {sleep_sec}s 후")
        time.sleep(sleep_sec)


if __name__ == "__main__":
    main()
