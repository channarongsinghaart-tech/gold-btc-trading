import os
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title='Gold & Bitcoin Trading Analyzer V4.4', page_icon='📈', layout='wide')

BASE = 'https://api.twelvedata.com'
ASSETS = {'Bitcoin BTC/USD': 'BTC/USD', 'Gold XAU/USD': 'XAU/USD'}
TF = {'5m': '5min', '15m': '15min', '1h': '1h', '4h': '4h', '1D': '1day'}

API_KEY = os.getenv('TWELVEDATA_API_KEY', '')
try:
    if not API_KEY and 'TWELVEDATA_API_KEY' in st.secrets:
        API_KEY = str(st.secrets['TWELVEDATA_API_KEY'])
except Exception:
    pass


def api_get(endpoint, params):
    if not API_KEY:
        return None, 'ยังไม่ได้ตั้ง TWELVEDATA_API_KEY'
    try:
        r = requests.get(f'{BASE}/{endpoint}', params=params, timeout=20)
        data = r.json()
    except Exception as exc:
        return None, f'เชื่อมต่อ Twelve Data ไม่สำเร็จ: {exc}'
    if r.status_code != 200 or ('code' in data and data.get('code', 200) >= 400):
        return None, data.get('message', 'Twelve Data API error')
    return data, None


@st.cache_data(ttl=45, show_spinner=False)
def get_ohlcv(symbol, interval, outputsize=500):
    data, err = api_get('time_series', {
        'symbol': symbol, 'interval': interval, 'outputsize': outputsize,
        'apikey': API_KEY, 'timezone': 'UTC'
    })
    if err:
        return pd.DataFrame(), err
    if not data or 'values' not in data:
        return pd.DataFrame(), (data or {}).get('message', 'ไม่มีข้อมูล')
    df = pd.DataFrame(data['values'])
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.sort_values('datetime').set_index('datetime')
    return df.dropna(subset=['open', 'high', 'low', 'close']), None


def indicators(df):
    d = df.copy()
    if d.empty:
        return d
    c, h, l = d.close, d.high, d.low
    for n in (20, 50, 200):
        d[f'EMA{n}'] = c.ewm(span=n, adjust=False).mean()
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d['ATR'] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d['range'] = h - l
    d['body'] = (c - d.open).abs()
    d['body_ratio'] = d.body / d['range'].replace(0, np.nan)
    d['upper_wick'] = h - d[['open', 'close']].max(axis=1)
    d['lower_wick'] = d[['open', 'close']].min(axis=1) - l
    if 'volume' in d.columns:
        d['VolMA20'] = d.volume.rolling(20).mean()
        d['VolRatio'] = d.volume / d.VolMA20
    else:
        d['VolMA20'] = np.nan
        d['VolRatio'] = np.nan
    return d


def closed_only(d):
    """Use only completed candles for analysis. Keep the latest row for display separately."""
    if d is None or len(d) < 3:
        return d
    return d.iloc[:-1].copy()


def structure(d, lookback=20):
    if d is None or len(d) < lookback + 8:
        return 'MIXED'
    recent = d.iloc[-6:]
    prior = d.iloc[-lookback:-6]
    rh, rl = float(recent.high.max()), float(recent.low.min())
    ph, pl = float(prior.high.max()), float(prior.low.min())
    if rh > ph and rl > pl:
        return 'HH_HL'
    if rh < ph and rl < pl:
        return 'LH_LL'
    x = d.iloc[-1]
    if x.EMA20 > x.EMA50 > x.EMA200:
        return 'BULL'
    if x.EMA20 < x.EMA50 < x.EMA200:
        return 'BEAR'
    return 'MIXED'


def trend(d):
    if d is None or len(d) < 205:
        return 'NEUTRAL', 50, 'MIXED'
    x = d.iloc[-1]
    stc = structure(d)
    bull = 0
    bear = 0
    bull += int(x.EMA20 > x.EMA50)
    bear += int(x.EMA20 < x.EMA50)
    bull += int(x.EMA50 > x.EMA200)
    bear += int(x.EMA50 < x.EMA200)
    bull += int(x.close > x.EMA20)
    bear += int(x.close < x.EMA20)
    slope = float(x.EMA20 - d.EMA20.iloc[-6])
    bull += int(slope > 0)
    bear += int(slope < 0)
    bull += 2 * int(stc in ('HH_HL', 'BULL'))
    bear += 2 * int(stc in ('LH_LL', 'BEAR'))
    diff = bull - bear
    bias = 'LONG' if diff > 0 else 'SHORT' if diff < 0 else 'NEUTRAL'
    score = int(np.clip(50 + abs(diff) * 10, 0, 100))
    return bias, score, stc


def range_info(d, n=48):
    x = d.tail(min(n, len(d)))
    hi = float(x.high.max())
    lo = float(x.low.min())
    span = max(hi - lo, 1e-9)
    price = float(d.close.iloc[-1])
    pos = (price - lo) / span
    zone = 'LOWER RANGE' if pos < 0.33 else 'UPPER RANGE' if pos > 0.67 else 'MID RANGE'
    return {'high': hi, 'low': lo, 'span': span, 'pos': pos, 'zone': zone}


def market_bias(d1, h4, h1):
    b1, s1, st1 = trend(d1)
    b4, s4, st4 = trend(h4)
    bH, sH, stH = trend(h1)
    # D1 is the primary direction. H4/H1 are allowed to be in a pullback state.
    if b1 in ('LONG', 'SHORT'):
        return b1, b1, b4, bH, s1, s4, sH, st1, st4, stH
    if b4 == bH and b4 in ('LONG', 'SHORT'):
        return b4, b4, b4, bH, s1, s4, sH, st1, st4, stH
    return 'NEUTRAL', b1, b4, bH, s1, s4, sH, st1, st4, stH


def m15_setup(d, direction):
    """Find a tradable setup on M15. Designed to be permissive enough to produce setups, not entries."""
    if d is None or len(d) < 80 or direction == 'NEUTRAL':
        return {'ok': False, 'score': 0, 'name': 'รอ M15 setup', 'type': 'NONE'}
    x = d.iloc[-1]
    prev = d.iloc[-2]
    atr = max(float(x.ATR), 1e-9)
    recent_hi = float(d.iloc[-21:-1].high.max())
    recent_lo = float(d.iloc[-21:-1].low.min())
    ema20, ema50 = float(x.EMA20), float(x.EMA50)
    dist20 = abs(float(x.close) - ema20) / atr

    bull_pull = (
        x.low <= ema20 + 0.35 * atr and x.close > ema20 and
        x.close > x.open and x.close >= x.low + 0.55 * max(float(x.range), 1e-9)
    )
    bear_pull = (
        x.high >= ema20 - 0.35 * atr and x.close < ema20 and
        x.close < x.open and x.close <= x.high - 0.55 * max(float(x.range), 1e-9)
    )
    bull_sweep = x.low < float(d.iloc[-8:-1].low.min()) and x.close > float(d.iloc[-8:-1].low.min()) and x.close > x.open
    bear_sweep = x.high > float(d.iloc[-8:-1].high.max()) and x.close < float(d.iloc[-8:-1].high.max()) and x.close < x.open
    bull_break = x.close > recent_hi and x.close > x.open and x.body_ratio >= 0.40
    bear_break = x.close < recent_lo and x.close < x.open and x.body_ratio >= 0.40

    names = []
    if direction == 'LONG':
        if bull_pull and ema20 >= ema50 * 0.998:
            names.append('M15 pullback')
        if bull_sweep:
            names.append('M15 sweep-reclaim')
        if bull_break:
            names.append('M15 breakout')
    else:
        if bear_pull and ema20 <= ema50 * 1.002:
            names.append('M15 pullback')
        if bear_sweep:
            names.append('M15 sweep-reject')
        if bear_break:
            names.append('M15 breakdown')

    # If the market is in a pullback but the exact candle has not yet triggered,
    # expose a "near setup" state instead of pretending there is an entry.
    near_pull = dist20 <= 0.90
    if not names and near_pull:
        return {'ok': False, 'near': True, 'score': 55, 'name': 'M15 ใกล้โซน setup', 'type': 'NEAR'}
    return {'ok': bool(names), 'near': False, 'score': min(100, 55 + 15 * len(names)),
            'name': ' + '.join(names) if names else 'รอ M15 pullback / breakout / sweep',
            'type': names[0] if names else 'NONE'}


def m5_trigger(d, direction):
    if d is None or len(d) < 40 or direction == 'NEUTRAL':
        return {'ok': False, 'score': 0, 'name': 'รอ M5 trigger', 'type': 'NONE'}
    x = d.iloc[-1]
    p = d.iloc[-2]
    atr = max(float(x.ATR), 1e-9)
    rg = max(float(x.range), 1e-9)
    vr = float(x.VolRatio) if np.isfinite(x.VolRatio) else 1.0
    bull = (
        (x.close > x.open and x.close > p.high and x.body_ratio >= 0.30) or
        (x.close > x.open and x.lower_wick >= 0.30 * rg and x.close >= x.low + 0.65 * rg) or
        (x.close > x.EMA20 and p.close <= p.EMA20 and x.close > x.open)
    )
    bear = (
        (x.close < x.open and x.close < p.low and x.body_ratio >= 0.30) or
        (x.close < x.open and x.upper_wick >= 0.30 * rg and x.close <= x.high - 0.65 * rg) or
        (x.close < x.EMA20 and p.close >= p.EMA20 and x.close < x.open)
    )
    if direction == 'LONG':
        score = 50 + (30 if bull else 0) + (10 if x.close > x.EMA20 else 0) + (10 if vr >= 1.05 else 0)
        return {'ok': score >= 70, 'score': int(min(100, score)),
                'name': 'M5 bullish trigger' if score >= 70 else 'รอ M5 กลับขึ้น',
                'type': 'LONG' if score >= 70 else 'NONE'}
    score = 50 + (30 if bear else 0) + (10 if x.close < x.EMA20 else 0) + (10 if vr >= 1.05 else 0)
    return {'ok': score >= 70, 'score': int(min(100, score)),
            'name': 'M5 bearish trigger' if score >= 70 else 'รอ M5 กลับลง',
            'type': 'SHORT' if score >= 70 else 'NONE'}


def make_plan(m15, m5, h1, h4, direction, mode):
    if direction == 'NEUTRAL':
        return None
    p15 = float(m15.close.iloc[-1])
    a15 = max(float(m15.ATR.iloc[-1]), 1e-9)
    p5 = float(m5.close.iloc[-1])
    a5 = max(float(m5.ATR.iloc[-1]), 1e-9)
    entry = p5 if mode == 'Short Hold' else p15

    if direction == 'LONG':
        if mode == 'Short Hold':
            swing = float(m5.tail(10).low.min())
            sl = swing - 0.35 * a5
            max_risk = 2.2 * a5
            if entry - sl > max_risk:
                return None
            risk = entry - sl
            tp1, tp2 = entry + 1.25 * risk, entry + 2.0 * risk
        else:
            swing = min(float(m15.tail(12).low.min()), float(h1.tail(8).low.min()))
            sl = swing - 0.40 * a15
            max_risk = 4.0 * a15
            if entry - sl > max_risk:
                return None
            risk = entry - sl
            tp1, tp2 = entry + 1.5 * risk, entry + 3.0 * risk
    else:
        if mode == 'Short Hold':
            swing = float(m5.tail(10).high.max())
            sl = swing + 0.35 * a5
            max_risk = 2.2 * a5
            if sl - entry > max_risk:
                return None
            risk = sl - entry
            tp1, tp2 = entry - 1.25 * risk, entry - 2.0 * risk
        else:
            swing = max(float(m15.tail(12).high.max()), float(h1.tail(8).high.max()))
            sl = swing + 0.40 * a15
            max_risk = 4.0 * a15
            if sl - entry > max_risk:
                return None
            risk = sl - entry
            tp1, tp2 = entry - 1.5 * risk, entry - 3.0 * risk
    if risk <= 0:
        return None
    return {'entry': float(entry), 'sl': float(sl), 'tp1': float(tp1), 'tp2': float(tp2),
            'risk': float(risk), 'rr': float(abs(tp2-entry)/risk)}


def evaluate(frames, mode):
    d1, h4, h1, m15, m5 = [frames[k] for k in ['1D', '4h', '1h', '15m', '5m']]
    primary, b1, b4, bh, s1, s4, sh, st1, st4, sth = market_bias(d1, h4, h1)
    setup = m15_setup(m15, primary)
    trigger = m5_trigger(m5, primary)
    r15 = range_info(m15, 48)

    # Long Hold tolerates an M5 counter-trend pullback. Short Hold needs M5 timing.
    h_alignment = 100 if b4 == primary and bh == primary else 75 if b4 == primary or bh == primary else 40
    range_score = 80 if (primary == 'LONG' and r15['zone'] == 'LOWER RANGE') or (primary == 'SHORT' and r15['zone'] == 'UPPER RANGE') else 60 if r15['zone'] == 'MID RANGE' else 35
    readiness = int(np.clip(0.35*s1 + 0.25*h_alignment + 0.25*setup['score'] + 0.15*trigger['score'], 0, 100))

    entry_ready = False
    if primary != 'NEUTRAL' and setup['ok']:
        if mode == 'Short Hold':
            entry_ready = trigger['ok'] and h_alignment >= 75
        else:
            # Long Hold: M15 is the setup; M5 is a timing bonus, not a hard veto.
            entry_ready = h_alignment >= 75 and (trigger['ok'] or setup['type'] in ('M15 breakout', 'M15 breakdown'))

    plan = make_plan(m15, m5, h1, h4, primary, mode) if entry_ready else None
    if entry_ready and plan is None:
        entry_ready = False

    if primary == 'NEUTRAL':
        status = 'WAIT'; reason = 'D1/H4/H1 ยังไม่ชัด'
    elif entry_ready:
        status = 'ENTRY READY'
        reason = f"{setup['name']} • {trigger['name']}"
    elif not setup['ok']:
        status = 'WAIT'; reason = 'รอ M15 pullback / breakout / sweep'
    elif mode == 'Short Hold' and not trigger['ok']:
        status = 'WAIT'; reason = 'M15 พร้อม แต่รอ M5 trigger'
    elif mode == 'Long Hold' and h_alignment < 75:
        status = 'WAIT'; reason = 'รอ H4/H1 alignment'
    else:
        status = 'WAIT'; reason = 'รอจังหวะเข้า'

    return {
        'status': status, 'direction': primary, 'trend_d1': s1,
        'h4_score': s4, 'h1_score': sh, 'alignment': h_alignment,
        'setup': setup, 'trigger': trigger, 'range': r15,
        'readiness': readiness, 'plan': plan, 'reason': reason,
        'state_d1': st1, 'state_h4': st4, 'state_h1': sth,
        'bias_h4': b4, 'bias_h1': bh, 'range_score': range_score
    }


def fmt(v):
    return '—' if v is None or not np.isfinite(v) else f'{v:,.2f}'


def render_card(result, title):
    ready = result['status'] == 'ENTRY READY'
    icon = '🟢' if ready and result['direction'] == 'LONG' else '🔴' if ready else '🟡'
    st.markdown(f'### {title}')
    st.markdown(f'**{icon} {result["status"]} — {result["direction"]}**')
    a, b, c = st.columns(3)
    a.metric('Trend D1', f"{result['trend_d1']}/100")
    b.metric('H4/H1 Structure', f"{result['alignment']}/100")
    c.metric('Setup Readiness', f"{result['readiness']}/100")
    st.caption(f"H4 {result['bias_h4']} • H1 {result['bias_h1']} • M15 {result['setup']['name']} • M5 {result['trigger']['name']}")
    st.info(f"สถานะ: {result['reason']}")
    if ready and result['plan']:
        p = result['plan']
        a, b, c, d = st.columns(4)
        a.metric('Entry', fmt(p['entry'])); b.metric('SL', fmt(p['sl']))
        c.metric('TP1', fmt(p['tp1'])); d.metric('TP2', fmt(p['tp2']))
        st.caption(f"M15 {result['range']['zone']} • R:R to TP2 ≈ {p['rr']:.2f}R")
    else:
        st.caption('ยังไม่แสดง Entry / SL / TP จนกว่าจะเกิด ENTRY READY')


st.title('Gold & Bitcoin Trading Analyzer — V4.4')
st.caption('D1 trend → H4/H1 structure & range → M15 setup → M5 entry trigger. Short Hold และ Long Hold แสดงพร้อมกัน')

with st.sidebar:
    st.header('ตั้งค่าการวิเคราะห์')
    asset_name = st.selectbox('สินทรัพย์', list(ASSETS.keys()), index=0)
    symbol = ASSETS[asset_name]
    chart_tf = st.selectbox('Timeframe กราฟ', list(TF.keys()), index=1)
    outputsize = st.slider('จำนวนแท่ง', 250, 800, 500, 50)
    auto = st.checkbox('Auto refresh', False)
    refresh = st.slider('รอบรีเฟรช (วินาที)', 30, 300, 60, 10)
    if st.button('รีเฟรชข้อมูลตอนนี้'):
        st.cache_data.clear(); st.rerun()
    st.divider(); st.caption('API: Twelve Data')

if auto:
    st.markdown(f'<meta http-equiv="refresh" content="{refresh}">', unsafe_allow_html=True)

frames = {}; errors = {}
for label in ['1D', '4h', '1h', '15m', '5m']:
    raw, err = get_ohlcv(symbol, TF[label], outputsize)
    if err:
        errors[label] = err
        frames[label] = pd.DataFrame()
    else:
        frames[label] = indicators(closed_only(raw))

chart_raw, chart_err = get_ohlcv(symbol, TF[chart_tf], outputsize)
chart = indicators(chart_raw) if not chart_raw.empty else pd.DataFrame()
if chart_err:
    errors[chart_tf] = chart_err
if errors:
    st.error(' | '.join(f'{k}: {v}' for k, v in errors.items()))
    st.stop()

price = float(chart.close.iloc[-1])
prev = float(chart.close.iloc[-2]) if len(chart) > 1 else price
pct = (price / prev - 1) * 100 if prev else 0
st.subheader(asset_name)
st.metric('Price', fmt(price), f'{pct:+.2f}%')

short = evaluate(frames, 'Short Hold')
long = evaluate(frames, 'Long Hold')

st.markdown('## สถานะการเทรด')
a, b = st.columns(2)
with a: render_card(short, 'Short Hold')
with b: render_card(long, 'Long Hold')

st.markdown('## โครงสร้างตลาด')
rows = []
for tf in ['1D', '4h', '1h', '15m', '5m']:
    d = frames[tf]
    b, s, stc = trend(d)
    rg = range_info(d, 48)
    rows.append({'TF': tf, 'Bias': b, 'Trend': s, 'Structure': stc, 'Range': rg['zone'], 'Range High': rg['high'], 'Range Low': rg['low']})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown(f'## กราฟ {chart_tf}')
fig = go.Figure()
fig.add_trace(go.Candlestick(x=chart.index, open=chart.open, high=chart.high, low=chart.low, close=chart.close, name='Price'))
for n in (20, 50, 200):
    fig.add_trace(go.Scatter(x=chart.index, y=chart[f'EMA{n}'], name=f'EMA{n}', mode='lines'))
fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

st.markdown('## หลักการของ V4.4')
st.write('D1 กำหนดทิศทางหลัก. H4/H1 ใช้ดูโครงสร้างและช่วงของราคา. M15 หา setup. M5 หา trigger. WAIT จะไม่แสดง Entry/SL/TP เพื่อไม่ให้สับสนกับสัญญาณที่พร้อมเข้า. Short Hold ต้องมี M5 trigger; Long Hold ใช้ M15 เป็น setup หลักและ M5 เป็นตัวช่วยจับจังหวะ.')
