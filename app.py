import os, time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Gold & BTC Live Analyzer V2", layout="wide")

API_KEY = os.getenv("TWELVEDATA_API_KEY", "")
BASE = "https://api.twelvedata.com"

ASSETS = {
    "Gold XAU/USD": "XAU/USD",
    "Bitcoin BTC/USD": "BTC/USD",
}
TF = {"5m":"5min", "15m":"15min", "1h":"1h", "4h":"4h", "1D":"1day"}

@st.cache_data(ttl=8)
def get_ohlcv(symbol, interval, outputsize=500):
    if not API_KEY:
        return pd.DataFrame(), "ยังไม่ได้ตั้ง TWELVEDATA_API_KEY"
    r = requests.get(
        f"{BASE}/time_series",
        params={"symbol":symbol, "interval":interval,
                "outputsize":outputsize, "apikey":API_KEY,
                "timezone":"UTC"},
        timeout=15)
    data = r.json()
    if "values" not in data:
        return pd.DataFrame(), data.get("message", "API error")
    df = pd.DataFrame(data["values"])
    for c in ["open","high","low","close","volume"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").set_index("datetime")
    return df.dropna(subset=["open","high","low","close"]), None

def add_indicators(df):
    d = df.copy()
    c, h, l = d.close, d.high, d.low
    d["EMA20"] = c.ewm(span=20, adjust=False).mean()
    d["EMA50"] = c.ewm(span=50, adjust=False).mean()
    d["EMA200"] = c.ewm(span=200, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
    d["RSI"] = 100 - 100/(1 + gain/loss.replace(0,np.nan))
    e12, e26 = c.ewm(span=12,adjust=False).mean(), c.ewm(span=26,adjust=False).mean()
    d["MACD"] = e12-e26
    d["MACDsig"] = d.MACD.ewm(span=9,adjust=False).mean()
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1/14,adjust=False).mean()
    d["VolMA20"] = d["volume"].rolling(20).mean() if "volume" in d else np.nan
    return d

def levels(d):
    recent = d.tail(80)
    resistance = recent.high.rolling(10, center=True).max().dropna().nlargest(5).mean()
    support = recent.low.rolling(10, center=True).min().dropna().nsmallest(5).mean()
    return float(support), float(resistance)

def signal(d):
    x=d.iloc[-1]; score=50; reasons=[]
    checks = [
        (x.close>x.EMA20, 8, "ราคาเหนือ EMA20"),
        (x.EMA20>x.EMA50, 10, "EMA20 > EMA50"),
        (x.EMA50>x.EMA200, 10, "EMA50 > EMA200"),
        (x.MACD>x.MACDsig, 8, "MACD bullish"),
    ]
    for ok, pts, txt in checks:
        score += pts if ok else -pts
        reasons.append(("+" if ok else "-")+" "+txt)
    if 55 <= x.RSI < 70: score += 7; reasons.append("+ RSI สนับสนุน momentum")
    elif x.RSI < 45: score -= 7; reasons.append("- RSI อ่อน")
    elif x.RSI >= 70: reasons.append("! RSI overbought")
    score=int(np.clip(score,0,100))
    return score, ("BUY" if score>=68 else "SELL" if score<=32 else "WAIT"), reasons

st.title("Gold & Bitcoin Live Trading Analyzer — V2")
st.caption("Streaming-ready architecture using Twelve Data REST; refresh/polling is used for candles. For true tick streaming, connect a WebSocket plan.")

with st.sidebar:
    asset = st.selectbox("สินทรัพย์", list(ASSETS))
    timeframe = st.selectbox("Timeframe", list(TF), index=1)
    refresh = st.slider("รีเฟรช (วินาที)", 5, 60, 10)
    st.markdown("**API:** Twelve Data")
    if API_KEY: st.success("API key loaded")
    else: st.warning("ตั้ง environment variable: TWELVEDATA_API_KEY")

symbol=ASSETS[asset]
df, err=get_ohlcv(symbol, TF[timeframe])
if err:
    st.error(err)
    st.code("export TWELVEDATA_API_KEY='YOUR_KEY'")
    st.stop()
if len(df)<220:
    st.warning("ข้อมูลยังไม่พอสำหรับ EMA200")
d=add_indicators(df)
score,sig,reasons=signal(d)
last=d.iloc[-1]; price=float(last.close)
prev=float(d.close.iloc[-2]); pct=(price/prev-1)*100
sup,res=levels(d)

a,b,c,e=st.columns(4)
a.metric("Live price",f"{price:,.4f}",f"{pct:+.2f}%")
b.metric("Signal",sig)
c.metric("Score",f"{score}/100")
e.metric("RSI",f"{last.RSI:.1f}")

fig=go.Figure(go.Candlestick(x=d.index,open=d.open,high=d.high,low=d.low,close=d.close,name=symbol))
for col in ["EMA20","EMA50","EMA200"]:
    fig.add_trace(go.Scatter(x=d.index,y=d[col],name=col,mode="lines"))
fig.add_hline(y=sup, annotation_text="Support")
fig.add_hline(y=res, annotation_text="Resistance")
fig.update_layout(height=650,xaxis_rangeslider_visible=False)
st.plotly_chart(fig,use_container_width=True)

l,r=st.columns(2)
with l:
    st.subheader("Signal engine")
    for x in reasons: st.write(x)
    st.write(f"Support: **{sup:,.4f}**")
    st.write(f"Resistance: **{res:,.4f}**")
with r:
    atr=float(last.ATR)
    st.subheader("Risk model")
    if sig=="BUY":
        sl=price-1.5*atr; tp1=price+1.5*atr; tp2=price+3*atr
    elif sig=="SELL":
        sl=price+1.5*atr; tp1=price-1.5*atr; tp2=price-3*atr
    else: sl=tp1=tp2=np.nan
    if sig!="WAIT":
        st.write(f"Entry: **{price:,.4f}**")
        st.write(f"SL: **{sl:,.4f}**")
        st.write(f"TP1: **{tp1:,.4f}**")
        st.write(f"TP2: **{tp2:,.4f}**")
    else:
        st.write("WAIT — setup ยังไม่ชัดเจน")

st.subheader("Multi-timeframe confirmation")
mtf=[]
for label in ["15m","1h","4h"]:
    x, er=get_ohlcv(symbol,TF[label])
    if not er and len(x)>=220:
        xd=add_indicators(x); sc,sg,_=signal(xd)
        mtf.append((label,sg,sc))
if mtf:
    st.dataframe(pd.DataFrame(mtf,columns=["Timeframe","Signal","Score"]),hide_index=True)
else:
    st.info("ต้องมี API และข้อมูลเพียงพอสำหรับ MTF")

st.caption("ข้อมูลตลาดภายนอกอาจมีข้อจำกัด/ความหน่วงตามแพ็กเกจผู้ให้บริการ. โปรแกรมนี้ไม่ส่งคำสั่งซื้อขายอัตโนมัติ.")
