import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Gold & BTC Trading Analyzer V3", page_icon="📈", layout="wide")

BASE = "https://api.twelvedata.com"
ASSETS = {
    "Gold XAU/USD": "XAU/USD",
    "Bitcoin BTC/USD": "BTC/USD",
}
TF = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1D": "1day"}

# Streamlit Cloud Secrets first; environment variable fallback for local use.
API_KEY = os.getenv("TWELVEDATA_API_KEY", "")
try:
    if not API_KEY and "TWELVEDATA_API_KEY" in st.secrets:
        API_KEY = str(st.secrets["TWELVEDATA_API_KEY"])
except Exception:
    pass


def api_get(endpoint, params):
    if not API_KEY:
        return None, "ยังไม่ได้ตั้ง TWELVEDATA_API_KEY"
    try:
        r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=20)
        data = r.json()
    except Exception as exc:
        return None, f"เชื่อมต่อ Twelve Data ไม่สำเร็จ: {exc}"
    if r.status_code != 200 or "code" in data and data.get("code", 200) >= 400:
        return None, data.get("message", "Twelve Data API error")
    return data, None


@st.cache_data(ttl=45, show_spinner=False)
def get_ohlcv(symbol, interval, outputsize=500):
    data, err = api_get(
        "time_series",
        {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": API_KEY,
            "timezone": "UTC",
        },
    )
    if err:
        return pd.DataFrame(), err
    if not data or "values" not in data:
        return pd.DataFrame(), (data or {}).get("message", "ไม่มีข้อมูล")

    df = pd.DataFrame(data["values"])
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").set_index("datetime")
    return df.dropna(subset=["open", "high", "low", "close"]), None


def add_indicators(df):
    d = df.copy()
    c, h, l = d["close"], d["high"], d["low"]

    for n in (20, 50, 200):
        d[f"EMA{n}"] = c.ewm(span=n, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))

    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    d["MACD"] = e12 - e26
    d["MACDsig"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACDhist"] = d["MACD"] - d["MACDsig"]

    tr = pd.concat(
        [(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    if "volume" in d.columns:
        d["VolMA20"] = d["volume"].rolling(20).mean()
        d["VolRatio"] = d["volume"] / d["VolMA20"]
    else:
        d["VolMA20"] = np.nan
        d["VolRatio"] = np.nan

    d["EMA20_slope"] = d["EMA20"].diff(5)
    d["range"] = d["high"] - d["low"]
    d["body"] = (d["close"] - d["open"]).abs()
    return d


def pivot_levels(d, lookback=120):
    x = d.tail(lookback)
    # Local swing levels; clustering is intentionally simple and robust.
    ph = x["high"].rolling(5, center=True).max()
    pl = x["low"].rolling(5, center=True).min()
    resistance_candidates = x.loc[x["high"] >= ph, "high"].dropna().tail(12)
    support_candidates = x.loc[x["low"] <= pl, "low"].dropna().tail(12)
    resistance = float(resistance_candidates.mean()) if len(resistance_candidates) else float(x["high"].tail(30).max())
    support = float(support_candidates.mean()) if len(support_candidates) else float(x["low"].tail(30).min())
    return support, resistance


def structure_flags(d):
    if len(d) < 30:
        return {"bull_breakout": False, "bear_breakdown": False, "bull_pullback": False, "bear_pullback": False,
                "bull_sweep": False, "bear_sweep": False, "bull_reversal": False, "bear_reversal": False}
    x = d.iloc[-1]
    prev = d.iloc[-2]
    prior = d.iloc[-21:-1]
    prior_high = float(prior["high"].max())
    prior_low = float(prior["low"].min())

    bull_breakout = x.close > prior_high and x.close > x.EMA20
    bear_breakdown = x.close < prior_low and x.close < x.EMA20

    bull_pullback = (
        x.close > x.EMA20 > x.EMA50
        and x.low <= x.EMA20 * 1.002
        and x.close > x.open
    )
    bear_pullback = (
        x.close < x.EMA20 < x.EMA50
        and x.high >= x.EMA20 * 0.998
        and x.close < x.open
    )

    recent_low = float(d.iloc[-8:-1]["low"].min())
    recent_high = float(d.iloc[-8:-1]["high"].max())
    bull_sweep = x.low < recent_low and x.close > recent_low and x.close > x.open
    bear_sweep = x.high > recent_high and x.close < recent_high and x.close < x.open

    bull_reversal = prev.RSI < 35 and x.RSI > prev.RSI and x.close > x.open
    bear_reversal = prev.RSI > 65 and x.RSI < prev.RSI and x.close < x.open

    return {
        "bull_breakout": bool(bull_breakout), "bear_breakdown": bool(bear_breakdown),
        "bull_pullback": bool(bull_pullback), "bear_pullback": bool(bear_pullback),
        "bull_sweep": bool(bull_sweep), "bear_sweep": bool(bear_sweep),
        "bull_reversal": bool(bull_reversal), "bear_reversal": bool(bear_reversal),
    }


def analyze(d):
    x = d.iloc[-1]
    score = 50.0
    reasons = []

    def add(points, text, positive=True):
        nonlocal score
        score += points
        reasons.append(("bull" if positive else "bear", ("+" if positive else "-") + " " + text, abs(points)))

    # Trend: strongest component.
    if x.close > x.EMA20:
        add(7, "ราคาเหนือ EMA20", True)
    else:
        add(-7, "ราคาใต้ EMA20", False)
    if x.EMA20 > x.EMA50:
        add(9, "EMA20 > EMA50", True)
    else:
        add(-9, "EMA20 < EMA50", False)
    if x.EMA50 > x.EMA200:
        add(9, "EMA50 > EMA200", True)
    else:
        add(-9, "EMA50 < EMA200", False)

    # Momentum.
    if x.MACD > x.MACDsig:
        add(7, "MACD bullish", True)
    else:
        add(-7, "MACD bearish", False)
    if 52 <= x.RSI <= 68:
        add(6, f"RSI {x.RSI:.1f} สนับสนุน momentum", True)
    elif 32 <= x.RSI < 48:
        add(-3, f"RSI {x.RSI:.1f} ยังอ่อน", False)
    elif x.RSI > 72:
        add(-2, f"RSI {x.RSI:.1f} สูงเกินไป", False)
    elif x.RSI < 28:
        add(2, f"RSI {x.RSI:.1f} อยู่เขต oversold", True)

    flags = structure_flags(d)
    for key, pts, text, positive in [
        ("bull_breakout", 10, "Bullish breakout", True),
        ("bear_breakdown", -10, "Bearish breakdown", False),
        ("bull_pullback", 7, "Bullish pullback ที่ EMA20", True),
        ("bear_pullback", -7, "Bearish pullback ที่ EMA20", False),
        ("bull_sweep", 6, "Bullish liquidity sweep", True),
        ("bear_sweep", -6, "Bearish liquidity sweep", False),
        ("bull_reversal", 6, "Bullish reversal momentum", True),
        ("bear_reversal", -6, "Bearish reversal momentum", False),
    ]:
        if flags[key]:
            add(pts, text, positive)

    # Volume confirmation, if supplied by the provider.
    if np.isfinite(x.get("VolRatio", np.nan)):
        if x.VolRatio >= 1.5 and x.close > x.open:
            add(5, f"Volume สูง {x.VolRatio:.1f}x ค่าเฉลี่ย", True)
        elif x.VolRatio >= 1.5 and x.close < x.open:
            add(-5, f"Volume สูง {x.VolRatio:.1f}x ค่าเฉลี่ย", False)

    score = int(np.clip(round(score), 0, 100))
    signal = "BUY" if score >= 68 else "SELL" if score <= 32 else "WAIT"
    # This is a model confidence score, not a statistical win probability.
    confidence = int(min(99, max(50, 50 + abs(score - 50) * 1.0)))
    return score, signal, confidence, reasons, flags


def risk_plan(d, signal, support, resistance):
    x = d.iloc[-1]
    price = float(x.close)
    atr = float(x.ATR)
    if not np.isfinite(atr) or atr <= 0 or signal == "WAIT":
        return None

    # Volatility-based initial stop, then nudge toward structure when sensible.
    if signal == "BUY":
        sl = price - 1.5 * atr
        if np.isfinite(support) and support < price:
            sl = min(sl, support - 0.25 * atr)
        risk = price - sl
        tp1 = price + 1.5 * risk
        tp2 = price + 2.5 * risk
    else:
        sl = price + 1.5 * atr
        if np.isfinite(resistance) and resistance > price:
            sl = max(sl, resistance + 0.25 * atr)
        risk = sl - price
        tp1 = price - 1.5 * risk
        tp2 = price - 2.5 * risk

    if risk <= 0:
        return None
    return {"entry": price, "sl": sl, "tp1": tp1, "tp2": tp2, "risk": risk, "rr1": 1.5, "rr2": 2.5}


def fmt(v):
    return f"{v:,.4f}"


st.title("Gold & Bitcoin Trading Analyzer — V3")
st.caption("Multi-timeframe technical/structure analyzer using Twelve Data REST. Confidence is a model score, not a guaranteed win probability.")

with st.sidebar:
    st.header("ตั้งค่าการวิเคราะห์")
    asset = st.selectbox("สินทรัพย์", list(ASSETS.keys()))
    timeframe = st.selectbox("Timeframe หลัก", list(TF.keys()), index=1)
    outputsize = st.select_slider("จำนวนแท่ง", options=[300, 500, 800], value=500)
    auto = st.checkbox("Auto refresh", value=False)
    refresh_seconds = st.slider("รอบรีเฟรช (วินาที)", 30, 300, 60, step=30)
    mtf_refresh = st.checkbox("วิเคราะห์ MTF ทุกครั้ง", value=False, help="เปิดเมื่ออยากยืนยัน 15m/1h/4h ทุกครั้ง แต่จะใช้ API credits มากขึ้น")
    if st.button("รีเฟรชข้อมูลตอนนี้", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.write("API: Twelve Data")
    st.success("API key loaded") if API_KEY else st.error("ยังไม่มี API key")

symbol = ASSETS[asset]

if auto:
    # Simple mobile-friendly refresh loop. Streamlit reruns the script after the delay.
    import time
    time.sleep(refresh_seconds)
    st.rerun()

df, err = get_ohlcv(symbol, TF[timeframe], outputsize)
if err:
    st.error(err)
    st.info("ตรวจสอบ Secrets: TWELVEDATA_API_KEY = \"YOUR_KEY\"")
    st.stop()

if len(df) < 220:
    st.warning("ข้อมูลน้อยกว่า 220 แท่ง — EMA200/โครงสร้างบางส่วนอาจยังไม่นิ่ง")

d = add_indicators(df)
score, sig, confidence, reasons, flags = analyze(d)
last = d.iloc[-1]
price = float(last.close)
prev = float(d.close.iloc[-2])
pct = (price / prev - 1) * 100
support, resistance = pivot_levels(d)
risk = risk_plan(d, sig, support, resistance)

# Top cards.
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Price", fmt(price), f"{pct:+.2f}%")
c2.metric("Signal", sig)
c3.metric("Score", f"{score}/100")
c4.metric("Confidence", f"{confidence}%")
c5.metric("RSI", f"{last.RSI:.1f}")

st.caption(f"แท่งล่าสุด: {last.name.strftime('%Y-%m-%d %H:%M UTC')} • {symbol} • {timeframe}")

# Chart.
fig = go.Figure()
fig.add_trace(go.Candlestick(x=d.index, open=d.open, high=d.high, low=d.low, close=d.close, name=symbol))
for col in ["EMA20", "EMA50", "EMA200"]:
    fig.add_trace(go.Scatter(x=d.index, y=d[col], name=col, mode="lines"))
fig.add_hline(y=support, annotation_text="Support", line_dash="dot")
fig.add_hline(y=resistance, annotation_text="Resistance", line_dash="dot")
fig.update_layout(height=620, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("เหตุผลของสัญญาณ")
    for direction, text, pts in reasons:
        st.write(text)
    st.write(f"Support: **{fmt(support)}**")
    st.write(f"Resistance: **{fmt(resistance)}**")

with right:
    st.subheader("Entry / SL / TP")
    if risk:
        st.write(f"Entry: **{fmt(risk['entry'])}**")
        st.write(f"Stop Loss: **{fmt(risk['sl'])}**")
        st.write(f"TP1: **{fmt(risk['tp1'])}**  (R:R 1:{risk['rr1']:.1f})")
        st.write(f"TP2: **{fmt(risk['tp2'])}**  (R:R 1:{risk['rr2']:.1f})")
        st.caption(f"ความเสี่ยงต่อหน่วยโดยประมาณ: {fmt(risk['risk'])}")
    else:
        st.write("WAIT — ยังไม่มี setup ที่เหมาะสมสำหรับวาง SL/TP")

st.subheader("Market Structure")
structure_labels = {
    "bull_breakout": "Bullish Breakout",
    "bear_breakdown": "Bearish Breakdown",
    "bull_pullback": "Bullish Pullback",
    "bear_pullback": "Bearish Pullback",
    "bull_sweep": "Bullish Liquidity Sweep",
    "bear_sweep": "Bearish Liquidity Sweep",
    "bull_reversal": "Bullish Reversal",
    "bear_reversal": "Bearish Reversal",
}
active = [label for key, label in structure_labels.items() if flags.get(key)]
st.write(" • ".join(active) if active else "ยังไม่พบ pattern โครงสร้างเด่นในแท่งล่าสุด")

st.subheader("Multi-Timeframe Confirmation")
mtf_frames = ["15m", "1h", "4h"]
mtf_rows = []
if mtf_refresh or "mtf_cache" not in st.session_state:
    mtf_cache = {}
    for label in mtf_frames:
        x, er = get_ohlcv(symbol, TF[label], 300)
        if not er and len(x) >= 220:
            xd = add_indicators(x)
            sc, sg, conf, _, _ = analyze(xd)
            mtf_cache[label] = {"Timeframe": label, "Signal": sg, "Score": sc, "Confidence": conf, "RSI": float(xd.iloc[-1].RSI)}
        else:
            mtf_cache[label] = {"Timeframe": label, "Signal": "ERROR", "Score": None, "Confidence": None, "RSI": None}
    st.session_state["mtf_cache"] = mtf_cache
else:
    mtf_cache = st.session_state["mtf_cache"]

for label in mtf_frames:
    if label in mtf_cache:
        mtf_rows.append(mtf_cache[label])
if mtf_rows:
    st.dataframe(pd.DataFrame(mtf_rows), hide_index=True, use_container_width=True)

valid_mtf = [r for r in mtf_rows if r["Signal"] in ("BUY", "SELL")]
if valid_mtf:
    buys = sum(r["Signal"] == "BUY" for r in valid_mtf)
    sells = sum(r["Signal"] == "SELL" for r in valid_mtf)
    if sig == "BUY" and buys >= 2:
        st.success("MTF ยืนยันฝั่ง BUY อย่างน้อย 2 จาก 3 timeframe")
    elif sig == "SELL" and sells >= 2:
        st.success("MTF ยืนยันฝั่ง SELL อย่างน้อย 2 จาก 3 timeframe")
    else:
        st.warning("สัญญาณ Timeframe หลักกับ MTF ยังไม่สอดคล้องกันทั้งหมด")

st.subheader("Indicator Snapshot")
st.dataframe(
    pd.DataFrame([
        ["EMA20", float(last.EMA20)], ["EMA50", float(last.EMA50)], ["EMA200", float(last.EMA200)],
        ["MACD", float(last.MACD)], ["MACD Signal", float(last.MACDsig)], ["ATR14", float(last.ATR)],
    ], columns=["Indicator", "Value"]),
    hide_index=True,
    use_container_width=True,
)

st.caption("V3 ยังเป็นระบบวิเคราะห์ ไม่ได้ส่งคำสั่งซื้อขายอัตโนมัติ และสัญญาณไม่ใช่คำแนะนำการลงทุน. ความถี่/ความสดของข้อมูลขึ้นกับแพ็กเกจ Twelve Data และ API credits.")
