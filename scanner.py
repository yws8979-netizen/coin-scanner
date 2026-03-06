import os, time
import requests
from datetime import datetime

BINANCE_FAPI = "https://fapi.binance.com"
TELEGRAM_API = "https://api.telegram.org"

BOT_TOKEN = os.getenv("T8664391200:AAGpet1y3CNiK1wp1RQQoWHnik4VsIcFUQU")
CHAT_ID   = os.getenv("T8013667300")

QUOTE = "USDT"
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "180"))
TOP_N = int(os.getenv("TOP_N", "10"))

MIN_QUOTE_VOL_15M = float(os.getenv("MIN_QUOTE_VOL_15M", "300000"))

FUNDING_SOFT_MAX = float(os.getenv("FUNDING_SOFT_MAX", "0.00012"))
FUNDING_HARD_MAX = float(os.getenv("FUNDING_HARD_MAX", "0.00025"))

W_PRICE_DROP = 22
W_OI_RISE = 22
W_OI_PER_DROP = 18
W_TAKER_RECOVER = 18
W_SHORT_BIAS = 10
W_FUNDING = 10

def http_get(url, params=None, timeout=10):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def tg_send(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("[WARN] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set.\n")
        print(text)
        return
    url = f"{TELEGRAM_API}/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "disable_web_page_preview": True}
    r = requests.post(url, json=payload, timeout=10)
    r.raise_for_status()

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def pct(a, b):
    if a == 0:
        return 0.0
    return (b - a) / a * 100.0

def get_usdt_perp_symbols():
    data = http_get(f"{BINANCE_FAPI}/fapi/v1/exchangeInfo")
    syms = []
    for s in data.get("symbols", []):
        if s.get("quoteAsset") != QUOTE:
            continue
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("status") != "TRADING":
            continue
        syms.append(s["symbol"])
    return syms

def get_klines(symbol, interval, limit):
    return http_get(f"{BINANCE_FAPI}/fapi/v1/klines", {
        "symbol": symbol, "interval": interval, "limit": limit
    })

def get_oi_hist(symbol, period, limit=30):
    return http_get(f"{BINANCE_FAPI}/futures/data/openInterestHist", {
        "symbol": symbol, "period": period, "limit": limit
    })

def get_taker_ratio(symbol, period, limit=30):
    return http_get(f"{BINANCE_FAPI}/futures/data/takerlongshortRatio", {
        "symbol": symbol, "period": period, "limit": limit
    })

def get_global_ls_ratio(symbol, period, limit=30):
    return http_get(f"{BINANCE_FAPI}/futures/data/globalLongShortAccountRatio", {
        "symbol": symbol, "period": period, "limit": limit
    })

def get_funding_rate(symbol):
    data = http_get(f"{BINANCE_FAPI}/fapi/v1/premiumIndex", {"symbol": symbol})
    return float(data["lastFundingRate"])

def score_symbol(symbol):
    try:
        k15 = get_klines(symbol, "15m", 6)
        if len(k15) < 5:
            return None, "no_klines"

        closes = [float(x[4]) for x in k15]
        quote_vols = [float(x[7]) for x in k15]

        if quote_vols[-1] < MIN_QUOTE_VOL_15M:
            return None, "low_liquidity"

        p_start_1h = closes[-5]
        p_end = closes[-1]
        price_chg_1h = pct(p_start_1h, p_end)

        if price_chg_1h > -1.0:
            return None, "no_drop"

        oi15 = get_oi_hist(symbol, "15m", limit=10)
        if len(oi15) < 5:
            return None, "no_oi"

        oi_vals = [float(x["sumOpenInterest"]) for x in oi15]
        oi_start_1h = oi_vals[-5]
        oi_end_1h = oi_vals[-1]
        oi_chg_1h = pct(oi_start_1h, oi_end_1h)

        if oi_chg_1h < 0.0:
            return None, "oi_not_rising"

        drop_abs = abs(price_chg_1h)
        oi_per_drop = (oi_chg_1h / drop_abs) if drop_abs > 0 else 0.0

        taker15 = get_taker_ratio(symbol, "15m", limit=6)
        if len(taker15) < 3:
            return None, "no_taker"

        ratios = [float(x["buySellRatio"]) for x in taker15]
        recent = ratios[-1]
        prev_avg = sum(ratios[-4:-1]) / 3.0
        taker_recover = recent - prev_avg

        gls15 = get_global_ls_ratio(symbol, "15m", limit=2)
        if len(gls15) < 1:
            return None, "no_gls"

        ls = float(gls15[-1]["longShortRatio"])
        short_bias = (1.0 - ls)

        funding = get_funding_rate(symbol)

        if funding >= FUNDING_HARD_MAX:
            return None, "funding_too_high"

        price_norm = clamp((abs(price_chg_1h) - 1.0) / (8.0 - 1.0), 0.0, 1.0)
        oi_norm = clamp(oi_chg_1h / 15.0, 0.0, 1.0)
        opd_norm = clamp(oi_per_drop / 3.0, 0.0, 1.0)
        tr_norm = clamp((taker_recover + 0.2) / 0.4, 0.0, 1.0)
        sb_norm = clamp(short_bias / 0.35, 0.0, 1.0)
        f_norm = 1.0 - clamp(funding / FUNDING_SOFT_MAX, 0.0, 1.0)

        score = (
            W_PRICE_DROP * price_norm +
            W_OI_RISE * oi_norm +
            W_OI_PER_DROP * opd_norm +
            W_TAKER_RECOVER * tr_norm +
            W_SHORT_BIAS * sb_norm +
            W_FUNDING * f_norm
        )

        if funding > 0 and recent > 1.5:
            score -= 5

        score = round(clamp(score, 0.0, 100.0), 1)

        details = {
            "symbol": symbol,
            "score": score,
            "price_chg_1h": round(price_chg_1h, 2),
            "oi_chg_1h": round(oi_chg_1h, 2),
            "oi_per_drop": round(oi_per_drop, 2),
            "taker_recent": round(recent, 3),
            "taker_recover": round(taker_recover, 3),
            "longShortRatio": round(ls, 3),
            "funding": funding,
            "quoteVol15m": round(quote_vols[-1], 0),
        }
        return score, details

    except Exception as e:
        return None, f"err:{type(e).__name__}"

def format_message(top_items):
    now_kst = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append(f"📡 숏스퀴즈 초입 후보 TOP{len(top_items)} (v3) — {now_kst}")
    lines.append("조건: 1h 눌림 + OI증가 + 테이커 회복 + 숏편향/펀딩중립 가산")
    lines.append("")

    for i, d in enumerate(top_items, 1):
        f_pct = d["funding"] * 100
        lines.append(
            f"{i:02d}) {d['symbol']}  점수 {d['score']}\n"
            f"   1h가격 {d['price_chg_1h']}% | 1hOI {d['oi_chg_1h']}% | (OI/낙폭) {d['oi_per_drop']}\n"
            f"   테이커(15m) {d['taker_recent']} (Δ{d['taker_recover']:+}) | L/S {d['longShortRatio']} | 펀딩 {f_pct:.4f}%\n"
            f"   Vol15m {int(d['quoteVol15m']):,}"
        )

    return "\n".join(lines)

def main_loop():
    syms = get_usdt_perp_symbols()
    tg_send(
        "✅ v3 스캐너 시작\n"
        f"- 심볼: {len(syms)}개\n"
        f"- 필터: quoteVol15m ≥ {int(MIN_QUOTE_VOL_15M):,} USDT\n"
        f"- 펀딩: soft≤{FUNDING_SOFT_MAX*100:.4f}% / hard<{FUNDING_HARD_MAX*100:.4f}%"
    )

    while True:
        t0 = time.time()
        results = []

        for s in syms:
            score, payload = score_symbol(s)
            if score is None:
                continue
            results.append(payload)

        results.sort(key=lambda x: x["score"], reverse=True)
        top = results[:TOP_N]

        if top:
            tg_send(format_message(top))
        else:
            tg_send("⚠️ 조건을 만족하는 후보가 없음")

        elapsed = time.time() - t0
        sleep_sec = max(5, SCAN_INTERVAL_SEC - int(elapsed))
        time.sleep(sleep_sec)

if __name__ == "__main__":
    main_loop()
