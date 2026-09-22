import os
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title='Gold & Bitcoin Trading Analyzer V4.5', page_icon='📈', layout='wide')

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
        d['VolRatio'] = d.volume / d.VolMA20.replace(0, np.nan)
    else:
        d['VolMA20'] = np.nan
        d['VolRatio'] = np.nan
    return d


def closed_only(d):
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


def tf_state(d):
    if d is None or len(d) < 205:
        return {'bias': 'NEUTRAL', 'score': 50, 'structure': 'MIXED', 'ema': 'MIXED', 'slope': 0.0}
    x = d.iloc[-1]
    stc = structure(d)
    ema_bull = x.EMA20 > x.EMA50 > x.EMA200
    ema_bear = x.EMA20 < x.EMA50 < x.EMA200
    slope = float(x.EMA20 - d.EMA20.iloc[-6])
    close_bull = x.close > x.EMA20
    close_bear = x.close < x.EMA20
    bull = int(ema_bull) + int(close_bull) + int(slope > 0) + 2 * int(stc in ('HH_HL', 'BULL'))
    bear = int(ema_bear) + int(close_bear) + int(slope < 0) + 2 * int(stc in ('LH_LL', 'BEAR'))
    diff = bull - bear
    bias = 'LONG' if diff > 0 else 'SHORT' if diff < 0 else 'NEUTRAL'
    score = int(np.clip(50 + abs(diff) * 10, 0, 100))
    ema = 'LONG' if ema_bull else 'SHORT' if ema_bear else 'MIXED'
    return {'bias': bias, 'score': score, 'structure': stc, 'ema': ema, 'slope': slope}


def range_info(d, n=48):
    x = d.tail(min(n, len(d)))
    hi, lo = float(x.high.max()), float(x.low.min())
    span = max(hi - lo, 1e-9)
    price = float(d.close.iloc[-1])
    pos = (price - lo) / span
    zone = 'LOWER RANGE' if pos < 0.33 else 'UPPER RANGE' if pos > 0.67 else 'MID RANGE'
    return {'high': hi, 'low': lo, 'span': span, 'pos': pos, 'zone': zone}


def macro_context(d1, h4, h1, m15):
    s1, s4, sH, s15 = [tf_state(x) for x in [d1, h4, h1, m15]]
    # Macro uses D1/H4. Tactical uses H1/M15. They are intentionally separate.
    if s1['bias'] == s4['bias'] and s1['bias'] in ('LONG', 'SHORT'):
        macro = s1['bias']
        macro_strength = int(round((s1['score'] + s4['score']) / 2))
    elif s1['bias'] in ('LONG', 'SHORT') and s4['bias'] == 'NEUTRAL':
        macro = s1['bias']; macro_strength = int(s1['score'] * 0.9)
    elif s4['bias'] in ('LONG', 'SHORT') and s1['bias'] == 'NEUTRAL':
        macro = s4['bias']; macro_strength = int(s4['score'] * 0.9)
    else:
        macro = 'NEUTRAL'; macro_strength = int((s1['score'] + s4['score']) / 2)

    if sH['bias'] == s15['bias'] and sH['bias'] in ('LONG', 'SHORT'):
        tactical = sH['bias']; tactical_strength = int(round((sH['score'] + s15['score']) / 2))
    elif s15['bias'] in ('LONG', 'SHORT'):
        tactical = s15['bias']; tactical_strength = s15['score']
    elif sH['bias'] in ('LONG', 'SHORT'):
        tactical = sH['bias']; tactical_strength = sH['score']
    else:
        tactical = 'NEUTRAL'; tactical_strength = int((sH['score'] + s15['score']) / 2)

    relation = 'ALIGNED' if macro == tactical and macro != 'NEUTRAL' else 'PULLBACK' if macro in ('LONG','SHORT') and tactical in ('LONG','SHORT') else 'MIXED'
    return {
        'd1': s1, 'h4': s4, 'h1': sH, 'm15': s15,
        'macro': macro, 'macro_strength': macro_strength,
        'tactical': tactical, 'tactical_strength': tactical_strength,
        'relation': relation,
    }


def m15_setup(d, tactical):
    if d is None or len(d) < 80 or tactical == 'NEUTRAL':
        return {'ok': False, 'near': False, 'score': 0, 'name': 'รอ M15 setup', 'type': 'NONE'}
    x, p = d.iloc[-1], d.iloc[-2]
    atr = max(float(x.ATR), 1e-9); rg = max(float(x.range), 1e-9)
    hi20 = float(d.iloc[-21:-1].high.max()); lo20 = float(d.iloc[-21:-1].low.min())
    ema20, ema50 = float(x.EMA20), float(x.EMA50)
    dist20 = abs(float(x.close) - ema20) / atr

    bull_pull = x.low <= ema20 + 0.55*atr and x.close > ema20 and x.close > x.open and x.close >= x.low + 0.55*rg
    bear_pull = x.high >= ema20 - 0.55*atr and x.close < ema20 and x.close < x.open and x.close <= x.high - 0.55*rg
    bull_sweep = x.low < float(d.iloc[-8:-1].low.min()) and x.close > float(d.iloc[-8:-1].low.min()) and x.close > x.open
    bear_sweep = x.high > float(d.iloc[-8:-1].high.max()) and x.close < float(d.iloc[-8:-1].high.max()) and x.close < x.open
    bull_break = x.close > hi20 and x.close > x.open and x.body_ratio >= 0.35
    bear_break = x.close < lo20 and x.close < x.open and x.body_ratio >= 0.35
    bull_cont = x.close > x.EMA20 and x.EMA20 >= x.EMA50 and x.low > float(d.iloc[-4:-1].low.min())
    bear_cont = x.close < x.EMA20 and x.EMA20 <= x.EMA50 and x.high < float(d.iloc[-4:-1].high.max())

    names = []
    if tactical == 'LONG':
        if bull_pull and ema20 >= ema50 * 0.997: names.append('M15 pullback')
        if bull_sweep: names.append('M15 sweep-reclaim')
        if bull_break: names.append('M15 breakout')
        if bull_cont: names.append('M15 continuation')
    else:
        if bear_pull and ema20 <= ema50 * 1.003: names.append('M15 pullback')
        if bear_sweep: names.append('M15 sweep-reject')
        if bear_break: names.append('M15 breakdown')
        if bear_cont: names.append('M15 continuation')

    near = dist20 <= 1.05 or (tactical == 'LONG' and x.close >= ema20) or (tactical == 'SHORT' and x.close <= ema20)
    if names:
        return {'ok': True, 'near': False, 'score': min(100, 60 + 10*len(names)), 'name': ' + '.join(names), 'type': names[0]}
    if near:
        return {'ok': False, 'near': True, 'score': 55, 'name': 'M15 near setup zone', 'type': 'NEAR'}
    return {'ok': False, 'near': False, 'score': 35, 'name': 'รอ M15 pullback / breakout / sweep / continuation', 'type': 'NONE'}


def m5_trigger(d, direction):
    if d is None or len(d) < 40 or direction == 'NEUTRAL':
        return {'ok': False, 'score': 0, 'name': 'รอ M5 trigger', 'type': 'NONE'}
    x, p = d.iloc[-1], d.iloc[-2]
    rg = max(float(x.range), 1e-9)
    vr = float(x.VolRatio) if np.isfinite(x.VolRatio) else 1.0
    bull_break = x.close > p.high and x.close > x.open and x.body_ratio >= 0.25
    bear_break = x.close < p.low and x.close < x.open and x.body_ratio >= 0.25
    bull_reclaim = x.close > x.EMA20 and p.close <= p.EMA20 and x.close > x.open
    bear_reclaim = x.close < x.EMA20 and p.close >= p.EMA20 and x.close < x.open
    bull_reject = x.lower_wick >= 0.25*rg and x.close > x.open and x.close >= x.low + 0.60*rg
    bear_reject = x.upper_wick >= 0.25*rg and x.close < x.open and x.close <= x.high - 0.60*rg
    if direction == 'LONG':
        points = 40*int(bull_break) + 35*int(bull_reclaim) + 25*int(bull_reject) + 10*int(vr >= 1.05)
        score = int(min(100, points + (30 if x.close > x.EMA20 else 0)))
        ok = score >= 65
        return {'ok': ok, 'score': score, 'name': 'M5 bullish trigger' if ok else 'รอ M5 กลับขึ้น', 'type': 'LONG' if ok else 'NONE'}
    points = 40*int(bear_break) + 35*int(bear_reclaim) + 25*int(bear_reject) + 10*int(vr >= 1.05)
    score = int(min(100, points + (30 if x.close < x.EMA20 else 0)))
    ok = score >= 65
    return {'ok': ok, 'score': score, 'name': 'M5 bearish trigger' if ok else 'รอ M5 กลับลง', 'type': 'SHORT' if ok else 'NONE'}


def make_plan(frames, direction, mode):
    m5, m15, h1, h4 = frames['5m'], frames['15m'], frames['1h'], frames['4h']
    if direction == 'NEUTRAL': return None
    entry = float(m5.close.iloc[-1])
    atr5 = max(float(m5.ATR.iloc[-1]), 1e-9)
    atr15 = max(float(m15.ATR.iloc[-1]), 1e-9)
    if mode == 'Short Hold':
        if direction == 'LONG':
            swing = float(m5.tail(12).low.min())
            sl = swing - 0.25*atr5
            risk = entry - sl
            if risk <= 0 or risk > 2.8*atr5: return None
            tp1, tp2 = entry + 1.2*risk, entry + 1.8*risk
        else:
            swing = float(m5.tail(12).high.max())
            sl = swing + 0.25*atr5
            risk = sl - entry
            if risk <= 0 or risk > 2.8*atr5: return None
            tp1, tp2 = entry - 1.2*risk, entry - 1.8*risk
    else:
        entry = float(m15.close.iloc[-1])
        if direction == 'LONG':
            swing = min(float(m15.tail(14).low.min()), float(h1.tail(10).low.min()))
            sl = swing - 0.30*atr15
            risk = entry - sl
            if risk <= 0 or risk > 5.0*atr15: return None
            tp1, tp2 = entry + 1.5*risk, entry + 3.0*risk
        else:
            swing = max(float(m15.tail(14).high.max()), float(h1.tail(10).high.max()))
            sl = swing + 0.30*atr15
            risk = sl - entry
            if risk <= 0 or risk > 5.0*atr15: return None
            tp1, tp2 = entry - 1.5*risk, entry - 3.0*risk
    return {'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'risk': risk, 'rr': abs(tp2-entry)/risk}


def evaluate(frames, mode):
    ctx = macro_context(frames['1D'], frames['4h'], frames['1h'], frames['15m'])
    m15s = ctx['tactical']
    setup = m15_setup(frames['15m'], m15s)
    trig = m5_trigger(frames['5m'], m15s)
    r15 = range_info(frames['15m'], 48)

    # Short Hold follows tactical direction. It may be counter-macro, but labels it explicitly.
    if mode == 'Short Hold':
        direction = m15s
        if direction == 'NEUTRAL':
            status = 'WAIT'; reason = 'H1/M15 ยังไม่ให้ทิศทาง'
        elif setup['ok'] and trig['ok']:
            status = 'ENTRY READY'; reason = f"{setup['name']} • {trig['name']}"
        elif setup['ok'] or setup['near']:
            status = 'PRE-ENTRY'; reason = f"{setup['name']} • {trig['name']}"
        else:
            status = 'WAIT'; reason = setup['name']
    else:
        # Long Hold follows macro direction only. A tactical counter-move is treated as a pullback, not a long entry.
        direction = ctx['macro']
        if direction == 'NEUTRAL':
            status = 'WAIT'; reason = 'D1/H4 ยังไม่ให้ Macro direction'
        elif ctx['tactical'] != direction:
            status = 'PRE-ENTRY' if setup['ok'] and setup['type'] in ('M15 pullback','M15 sweep-reclaim','M15 sweep-reject') else 'WAIT'
            reason = f"Macro {direction} • Tactical {ctx['tactical']} = pullback/rebound ภายใน Macro"
        elif setup['ok']:
            status = 'ENTRY READY' if trig['ok'] else 'PRE-ENTRY'
            reason = f"{setup['name']} • {trig['name']}"
        elif setup['near']:
            status = 'PRE-ENTRY'; reason = 'M15 ใกล้ setup zone'
        else:
            status = 'WAIT'; reason = setup['name']

    # A plan exists only for ENTRY READY. No premature Entry/SL/TP display.
    plan = make_plan(frames, direction, mode) if status == 'ENTRY READY' else None
    if status == 'ENTRY READY' and plan is None:
        status = 'PRE-ENTRY'
        reason = 'Setup พร้อม แต่ระยะ SL จากโครงสร้างกว้างเกินไป — รอราคาจัดโครงสร้างใหม่'

    relation = 'ALIGNED' if ctx['macro'] == ctx['tactical'] and ctx['macro'] != 'NEUTRAL' else 'COUNTER-MACRO' if ctx['macro'] != 'NEUTRAL' and ctx['tactical'] != 'NEUTRAL' else 'MIXED'
    readiness = int(np.clip(0.35*ctx['macro_strength'] + 0.30*ctx['tactical_strength'] + 0.20*setup['score'] + 0.15*trig['score'], 0, 100))
    return {
        'status': status, 'direction': direction, 'macro': ctx['macro'], 'tactical': ctx['tactical'],
        'macro_strength': ctx['macro_strength'], 'tactical_strength': ctx['tactical_strength'],
        'relation': relation, 'trend_d1': ctx['d1']['score'], 'h4_score': ctx['h4']['score'], 'h1_score': ctx['h1']['score'],
        'alignment': int(round((ctx['h4']['score'] + ctx['h1']['score'])/2)),
        'setup': setup, 'trigger': trig, 'range': r15, 'readiness': readiness, 'plan': plan, 'reason': reason,
        'states': ctx,
    }


def fmt(v):
    return '—' if v is None or not np.isfinite(v) else f'{v:,.2f}'


def render_card(result, title):
    ready = result['status'] == 'ENTRY READY'
    pre = result['status'] == 'PRE-ENTRY'
    icon = '🟢' if ready and result['direction'] == 'LONG' else '🔴' if ready else '🟡'
    st.markdown(f'### {title}')
    st.markdown(f'**{icon} {result["status"]} — {result["direction"]}**')
    a,b,c = st.columns(3)
    a.metric('Macro D1/H4', f"{result['macro']} {result['macro_strength']}/100")
    b.metric('Tactical H1/M15', f"{result['tactical']} {result['tactical_strength']}/100")
    c.metric('Setup Readiness', f"{result['readiness']}/100")
    st.caption(f"Macro {result['macro']} • Tactical {result['tactical']} • {result['relation']} • M15 {result['setup']['name']} • M5 {result['trigger']['name']}")
    if pre:
        st.info(f"PRE-ENTRY: {result['reason']}")
    elif ready:
        st.success(f"ENTRY READY: {result['reason']}")
    else:
        st.info(f"WAIT: {result['reason']}")
    if ready and result['plan']:
        p = result['plan']
        a,b,c,d = st.columns(4)
        a.metric('Entry', fmt(p['entry'])); b.metric('SL', fmt(p['sl'])); c.metric('TP1', fmt(p['tp1'])); d.metric('TP2', fmt(p['tp2']))
        st.caption(f"R:R to TP2 ≈ {p['rr']:.2f}R")
    else:
        st.caption('ยังไม่แสดง Entry / SL / TP จนกว่าจะเกิด ENTRY READY')


st.title('Gold & Bitcoin Trading Analyzer — V4.5')
st.caption('D1 Macro → H4 Macro structure → H1/M15 Tactical → M5 trigger. Short Hold และ Long Hold แสดงพร้อมกัน')

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

frames, errors = {}, {}
for label in ['1D','4h','1h','15m','5m']:
    raw, err = get_ohlcv(symbol, TF[label], outputsize)
    if err:
        errors[label] = err; frames[label] = pd.DataFrame()
    else:
        frames[label] = indicators(closed_only(raw))

chart_raw, chart_err = get_ohlcv(symbol, TF[chart_tf], outputsize)
chart = indicators(chart_raw) if not chart_raw.empty else pd.DataFrame()
if chart_err: errors[chart_tf] = chart_err
if errors:
    st.error(' | '.join(f'{k}: {v}' for k,v in errors.items())); st.stop()

price = float(chart.close.iloc[-1]); prev = float(chart.close.iloc[-2]) if len(chart)>1 else price
pct = (price/prev-1)*100 if prev else 0
st.subheader(asset_name); st.metric('Price', fmt(price), f'{pct:+.2f}%')

short = evaluate(frames, 'Short Hold'); long = evaluate(frames, 'Long Hold')
st.markdown('## สถานะการเทรด')
a,b = st.columns(2)
with a: render_card(short, 'Short Hold')
with b: render_card(long, 'Long Hold')

st.markdown('## โครงสร้างตลาด')
rows=[]
for tf in ['1D','4h','1h','15m','5m']:
    d=frames[tf]; s=tf_state(d); rg=range_info(d,48)
    rows.append({'TF':tf,'Bias':s['bias'],'Trend':s['score'],'Structure':s['structure'],'Range':rg['zone'],'Range High':rg['high'],'Range Low':rg['low']})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown(f'## กราฟ {chart_tf}')
fig=go.Figure(); fig.add_trace(go.Candlestick(x=chart.index,open=chart.open,high=chart.high,low=chart.low,close=chart.close,name='Price'))
for n in (20,50,200): fig.add_trace(go.Scatter(x=chart.index,y=chart[f'EMA{n}'],name=f'EMA{n}',mode='lines'))
fig.update_layout(height=520,xaxis_rangeslider_visible=False,margin=dict(l=10,r=10,t=30,b=10)); st.plotly_chart(fig,use_container_width=True)

st.markdown('## หลักการของ V4.5')
st.write('D1/H4 = Macro direction. H1/M15 = Tactical direction และ setup. M5 = entry trigger. Short Hold follows tactical direction and may be counter-macro; Long Hold follows macro direction and treats H1/M15 counter-moves as pullbacks. สถานะมี WAIT → PRE-ENTRY → ENTRY READY และจะแสดง Entry/SL/TP เฉพาะ ENTRY READY เท่านั้น. ใช้เฉพาะแท่งที่ปิดแล้วในการวิเคราะห์.')
