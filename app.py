import os
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title='Gold & Bitcoin Trading Analyzer V4.7', page_icon='📈', layout='wide')

BASE = 'https://api.twelvedata.com'
ASSETS = {'Bitcoin BTC/USD': 'BTC/USD', 'Gold XAU/USD': 'XAU/USD'}
TF = {'5m': '5min', '15m': '15min', '1h': '1h', '4h': '4h', '1D': '1day'}

# Free-plan API keys are rate-limited (both requests/minute and requests/day).
# These knobs keep the app usable on a free key: fewer requests per load, spaced
# out client-side, cached longer, and with a fallback to the last good data
# instead of a hard failure when a call is throttled.
MIN_CALL_INTERVAL = 1.1      # seconds between outgoing API calls (client-side pacing)
CACHE_TTL = 60                # seconds; also the practical floor for auto-refresh
_last_call_ts = 0.0
_LAST_GOOD = {}                # process-level fallback cache: {(symbol, interval): df}

API_KEY = os.getenv('TWELVEDATA_API_KEY', '')
try:
    if not API_KEY and 'TWELVEDATA_API_KEY' in st.secrets:
        API_KEY = str(st.secrets['TWELVEDATA_API_KEY'])
except Exception:
    pass


def _pace_requests():
    """Client-side spacing so a single page load doesn't burst all requests
    at once and trip a free-plan per-minute rate limit."""
    global _last_call_ts
    now = time.monotonic()
    wait = MIN_CALL_INTERVAL - (now - _last_call_ts)
    if wait > 0:
        time.sleep(wait)
    _last_call_ts = time.monotonic()


def api_get(endpoint, params):
    if not API_KEY:
        return None, 'ยังไม่ได้ตั้ง TWELVEDATA_API_KEY'
    last_err = 'Twelve Data API error'
    for attempt in range(2):  # one retry on rate-limit/transient failure
        _pace_requests()
        try:
            r = requests.get(f'{BASE}/{endpoint}', params=params, timeout=20)
            data = r.json()
        except Exception as exc:
            last_err = f'เชื่อมต่อ Twelve Data ไม่สำเร็จ: {exc}'
            time.sleep(1.5)
            continue
        if r.status_code == 429:
            last_err = 'Twelve Data rate limit (free plan) — คำขอถี่เกินไป'
            time.sleep(3.0)
            continue
        if r.status_code != 200 or ('code' in data and data.get('code', 200) >= 400):
            return None, data.get('message', 'Twelve Data API error')
        return data, None
    return None, last_err


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _fetch_ohlcv(symbol, interval, outputsize):
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


def get_ohlcv(symbol, interval, outputsize=500):
    """Wraps the cached fetch with a process-level fallback: if the live call
    fails (rate limit, network hiccup), reuse the last successful pull for
    this symbol/interval instead of taking the whole app down."""
    key = (symbol, interval)
    df, err = _fetch_ohlcv(symbol, interval, outputsize)
    if err or df.empty:
        cached = _LAST_GOOD.get(key)
        if cached is not None and not cached.empty:
            return cached, f'{err or "ไม่มีข้อมูลใหม่"} (ใช้ข้อมูลล่าสุดที่แคชไว้)'
        return df, err
    _LAST_GOOD[key] = df
    return df, None


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

    return {
        'd1': s1, 'h4': s4, 'h1': sH, 'm15': s15,
        'macro': macro, 'macro_strength': macro_strength,
        'tactical': tactical, 'tactical_strength': tactical_strength,
    }


def _m15_candidate(row, tactical, d):
    atr = max(float(row.ATR), 1e-9)
    rg = max(float(row.range), 1e-9)
    ema20, ema50 = float(row.EMA20), float(row.EMA50)
    hi20 = float(d.iloc[-21:-1].high.max())
    lo20 = float(d.iloc[-21:-1].low.min())
    recent_low = float(d.iloc[-8:-1].low.min())
    recent_high = float(d.iloc[-8:-1].high.max())

    bull_pull = row.low <= ema20 + 0.75*atr and row.close >= ema20 and row.close > row.open and row.close >= row.low + 0.50*rg
    bear_pull = row.high >= ema20 - 0.75*atr and row.close <= ema20 and row.close < row.open and row.close <= row.high - 0.50*rg
    bull_sweep = row.low < recent_low and row.close > recent_low and row.close > row.open
    bear_sweep = row.high > recent_high and row.close < recent_high and row.close < row.open
    bull_break = row.close > hi20 and row.close > row.open and row.body_ratio >= 0.30
    bear_break = row.close < lo20 and row.close < row.open and row.body_ratio >= 0.30
    bull_cont = row.close > ema20 and ema20 >= ema50*0.998 and row.low > float(d.iloc[-4:-1].low.min())
    bear_cont = row.close < ema20 and ema20 <= ema50*1.002 and row.high < float(d.iloc[-4:-1].high.max())

    if tactical == 'LONG':
        names=[]
        if bull_pull: names.append('M15 pullback')
        if bull_sweep: names.append('M15 sweep-reclaim')
        if bull_break: names.append('M15 breakout')
        if bull_cont: names.append('M15 continuation')
    else:
        names=[]
        if bear_pull: names.append('M15 pullback')
        if bear_sweep: names.append('M15 sweep-reject')
        if bear_break: names.append('M15 breakdown')
        if bear_cont: names.append('M15 continuation')
    return names


def m15_setup(d, tactical):
    if d is None or len(d) < 80 or tactical == 'NEUTRAL':
        return {'ok': False, 'near': False, 'score': 0, 'name': 'รอ M15 setup', 'type': 'NONE'}

    # Look at the last 3 CLOSED candles. This avoids requiring the exact setup to occur on one candle only.
    candidates=[]
    for off in (0,1,2):
        sub=d if off==0 else d.iloc[:-off]
        if len(sub) < 30: continue
        names=_m15_candidate(sub.iloc[-1], tactical, sub)
        for n in names:
            candidates.append((off,n))
    if candidates:
        # Prefer the newest qualifying setup, then the strongest multi-pattern candle.
        newest=min(o for o,_ in candidates)
        names=[]
        for off,n in candidates:
            if off==newest and n not in names: names.append(n)
        return {'ok': True, 'near': False, 'score': min(100, 62 + 8*len(names)), 'name': ' + '.join(names), 'type': names[0]}

    x=d.iloc[-1]; atr=max(float(x.ATR),1e-9)
    dist20=abs(float(x.close)-float(x.EMA20))/atr
    near=dist20 <= 1.25
    return {'ok': False, 'near': near, 'score': 52 if near else 35,
            'name': 'M15 ใกล้ setup zone' if near else 'รอ M15 pullback / breakout / sweep / continuation', 'type':'NEAR' if near else 'NONE'}


def m5_trigger(d, direction):
    if d is None or len(d) < 50 or direction == 'NEUTRAL':
        return {'ok': False, 'score': 0, 'name': 'รอ M5 trigger', 'type': 'NONE'}

    # Evaluate the last 3 closed candles so a valid trigger is not missed between
    # refreshes. Like m15_setup, prefer the NEWEST candle that clears the ok
    # threshold (55) rather than whichever of the 3 scores highest — otherwise a
    # trigger from 1-2 bars ago can keep firing after the latest candle has
    # already invalidated it.
    candidates=[]
    for off in (0,1,2):
        sub=d if off==0 else d.iloc[:-off]
        if len(sub) < 25: continue
        x,p=sub.iloc[-1],sub.iloc[-2]
        rg=max(float(x.range),1e-9)
        vr=float(x.VolRatio) if np.isfinite(x.VolRatio) else 1.0
        bull_break=x.close>p.high and x.close>x.open and x.body_ratio>=0.20
        bear_break=x.close<p.low and x.close<x.open and x.body_ratio>=0.20
        bull_reclaim=x.close>x.EMA20 and p.close<=p.EMA20 and x.close>x.open
        bear_reclaim=x.close<x.EMA20 and p.close>=p.EMA20 and x.close<x.open
        bull_reject=x.lower_wick>=0.20*rg and x.close>x.open and x.close>=x.low+0.55*rg
        bear_reject=x.upper_wick>=0.20*rg and x.close<x.open and x.close<=x.high-0.55*rg
        if direction=='LONG':
            score=min(100, 40*int(bull_break)+35*int(bull_reclaim)+25*int(bull_reject)+10*int(vr>=1.0)+20*int(x.close>x.EMA20))
            name='M5 bullish trigger'
        else:
            score=min(100, 40*int(bear_break)+35*int(bear_reclaim)+25*int(bear_reject)+10*int(vr>=1.0)+20*int(x.close<x.EMA20))
            name='M5 bearish trigger'
        candidates.append((off,score,name))
    if not candidates:
        return {'ok':False,'score':0,'name':'รอ M5 trigger','type':'NONE'}
    qualifying=[c for c in candidates if c[1]>=55]
    if qualifying:
        off,score,name=min(qualifying, key=lambda c: c[0])  # newest (smallest off) that qualifies
        return {'ok':True,'score':int(score),'name':name,'type':direction}
    off,score,name=max(candidates, key=lambda c: c[1])  # best near-miss, for messaging only
    return {'ok':False,'score':int(score),'name':'รอ M5 กลับขึ้น' if direction=='LONG' else 'รอ M5 กลับลง','type':'NONE'}


def scan_signals(d, threshold=55):
    """View-only chart overlay: scan every closed bar in d for a BUY/SELL
    trigger using the same per-bar scoring rules as m5_trigger. Unlike the
    Short Hold / Long Hold cards above, this does NOT check macro/tactical
    alignment or place any order — it just marks where the raw candle
    pattern fired on this timeframe, similar to a signal-marker chart."""
    buys, sells = [], []
    if d is None or len(d) < 2:
        return buys, sells
    for i in range(1, len(d)):
        x, p = d.iloc[i], d.iloc[i-1]
        rg = max(float(x.range), 1e-9)
        vr = float(x.VolRatio) if np.isfinite(x.VolRatio) else 1.0
        bull_break=x.close>p.high and x.close>x.open and x.body_ratio>=0.20
        bear_break=x.close<p.low and x.close<x.open and x.body_ratio>=0.20
        bull_reclaim=x.close>x.EMA20 and p.close<=p.EMA20 and x.close>x.open
        bear_reclaim=x.close<x.EMA20 and p.close>=p.EMA20 and x.close<x.open
        bull_reject=x.lower_wick>=0.20*rg and x.close>x.open and x.close>=x.low+0.55*rg
        bear_reject=x.upper_wick>=0.20*rg and x.close<x.open and x.close<=x.high-0.55*rg
        long_score=min(100, 40*int(bull_break)+35*int(bull_reclaim)+25*int(bull_reject)+10*int(vr>=1.0)+20*int(x.close>x.EMA20))
        short_score=min(100, 40*int(bear_break)+35*int(bear_reclaim)+25*int(bear_reject)+10*int(vr>=1.0)+20*int(x.close<x.EMA20))
        if long_score>=threshold:
            buys.append({'time': x.name, 'price': float(x.low), 'score': int(long_score)})
        if short_score>=threshold:
            sells.append({'time': x.name, 'price': float(x.high), 'score': int(short_score)})
    return buys, sells


def range_location(h4,h1,direction):
    if direction not in ('LONG','SHORT'):
        return {'score':50,'zone':'MIXED','reason':'ยังไม่มี direction'}
    r4=range_info(h4,48); r1=range_info(h1,48)
    zones=[r4['zone'],r1['zone']]
    if direction=='LONG':
        if zones.count('LOWER RANGE')==2: return {'score':90,'zone':'LOWER RANGE','reason':'H4/H1 อยู่โซนล่าง เหมาะกับการหาจังหวะ Long'}
        if 'LOWER RANGE' in zones: return {'score':75,'zone':'LOWER/MID','reason':'มีอย่างน้อยหนึ่ง TF อยู่โซนล่าง'}
        if zones.count('UPPER RANGE')==2: return {'score':30,'zone':'UPPER RANGE','reason':'ราคาอยู่โซนบนของ H4/H1 ไม่เหมาะกับการไล่ Long'}
        return {'score':55,'zone':'MID RANGE','reason':'H4/H1 อยู่กลาง range'}
    else:
        if zones.count('UPPER RANGE')==2: return {'score':90,'zone':'UPPER RANGE','reason':'H4/H1 อยู่โซนบน เหมาะกับการหาจังหวะ Short'}
        if 'UPPER RANGE' in zones: return {'score':75,'zone':'UPPER/MID','reason':'มีอย่างน้อยหนึ่ง TF อยู่โซนบน'}
        if zones.count('LOWER RANGE')==2: return {'score':30,'zone':'LOWER RANGE','reason':'ราคาอยู่โซนล่าง ไม่เหมาะกับการไล่ Short'}
        return {'score':55,'zone':'MID RANGE','reason':'H4/H1 อยู่กลาง range'}

def make_plan(frames, direction, mode):
    m5, m15, h1, h4 = frames['5m'], frames['15m'], frames['1h'], frames['4h']
    if direction == 'NEUTRAL': return None
    entry = float(m5.close.iloc[-1])
    atr5 = max(float(m5.ATR.iloc[-1]), 1e-9)
    atr15 = max(float(m15.ATR.iloc[-1]), 1e-9)
    # Minimum stop distance as a multiple of ATR. Without this floor, a
    # sweep/reclaim trigger (where entry sits right at the swing point by
    # construction) can produce a risk of just the 0.25/0.30*ATR buffer —
    # too tight for normal noise on a volatile instrument. The structural
    # (swing-based) stop is still used whenever it's already wider than this.
    MIN_RISK_ATR_MULT_5M = 1.0
    MIN_RISK_ATR_MULT_15M = 1.2
    if mode == 'Short Hold':
        min_risk, max_risk = MIN_RISK_ATR_MULT_5M*atr5, 2.8*atr5
        if direction == 'LONG':
            swing = float(m5.tail(12).low.min())
            sl = swing - 0.25*atr5
            risk = entry - sl
            if 0 < risk < min_risk:
                risk = min_risk; sl = entry - risk
            if risk <= 0 or risk > max_risk: return None
            tp1, tp2 = entry + 1.2*risk, entry + 1.8*risk
        else:
            swing = float(m5.tail(12).high.max())
            sl = swing + 0.25*atr5
            risk = sl - entry
            if 0 < risk < min_risk:
                risk = min_risk; sl = entry + risk
            if risk <= 0 or risk > max_risk: return None
            tp1, tp2 = entry - 1.2*risk, entry - 1.8*risk
    else:
        entry = float(m15.close.iloc[-1])
        min_risk, max_risk = MIN_RISK_ATR_MULT_15M*atr15, 5.0*atr15
        if direction == 'LONG':
            swing = min(float(m15.tail(14).low.min()), float(h1.tail(10).low.min()))
            sl = swing - 0.30*atr15
            risk = entry - sl
            if 0 < risk < min_risk:
                risk = min_risk; sl = entry - risk
            if risk <= 0 or risk > max_risk: return None
            tp1, tp2 = entry + 1.5*risk, entry + 3.0*risk
        else:
            swing = max(float(m15.tail(14).high.max()), float(h1.tail(10).high.max()))
            sl = swing + 0.30*atr15
            risk = sl - entry
            if 0 < risk < min_risk:
                risk = min_risk; sl = entry + risk
            if risk <= 0 or risk > max_risk: return None
            tp1, tp2 = entry - 1.5*risk, entry - 3.0*risk
    return {'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'risk': risk, 'rr': abs(tp2-entry)/risk}


def evaluate(frames, mode):
    ctx = macro_context(frames['1D'], frames['4h'], frames['1h'], frames['15m'])
    # Hierarchy: D1/H4 = macro, H1/M15 = tactical, M15 = setup, M5 = trigger.
    tactical = ctx['tactical']
    macro = ctx['macro']

    # Short Hold follows tactical direction. It may trade against the macro trend,
    # but that case is explicitly classified as COUNTER-MACRO rather than a normal trend entry.
    short_dir = tactical
    short_loc = range_location(frames['4h'], frames['1h'], short_dir)
    short_setup = m15_setup(frames['15m'], short_dir)
    short_trig = m5_trigger(frames['5m'], short_dir)

    # Long Hold follows macro direction. A tactical counter-move is a pullback/rebound
    # and can never be labeled ENTRY READY until tactical direction realigns with macro.
    long_loc = range_location(frames['4h'], frames['1h'], macro)
    long_setup = m15_setup(frames['15m'], macro)
    long_trig = m5_trigger(frames['5m'], macro)

    aligned = macro in ('LONG','SHORT') and tactical == macro
    counter_macro = macro in ('LONG','SHORT') and tactical in ('LONG','SHORT') and tactical != macro

    if mode == 'Short Hold':
        direction = short_dir
        setup, trig, loc = short_setup, short_trig, short_loc
        if direction == 'NEUTRAL':
            status = 'WAIT'; reason = 'H1/M15 ยังไม่ให้ tactical direction'; entry_class = 'WAIT'
        elif short_setup['ok'] and short_trig['ok']:
            status = 'ENTRY READY'
            entry_class = 'TREND ENTRY' if aligned else 'COUNTER-MACRO ENTRY' if counter_macro else 'TACTICAL ENTRY'
            prefix = 'ตาม Macro' if aligned else 'สวน Macro ระยะสั้น' if counter_macro else 'Macro ยังไม่ชัด'
            reason = f"{prefix} • {short_setup['name']} • {short_trig['name']} • {short_loc['zone']}"
        elif short_setup['ok'] or short_setup['near']:
            status = 'PRE-ENTRY'
            entry_class = 'COUNTER-MACRO WATCH' if counter_macro else 'TREND WATCH' if aligned else 'TACTICAL WATCH'
            reason = f"{short_setup['name']} • {short_trig['name']} • {short_loc['reason']}"
        else:
            status = 'WAIT'; entry_class = 'WAIT'; reason = short_setup['name']
    else:
        direction = macro
        setup, trig, loc = long_setup, long_trig, long_loc
        if direction == 'NEUTRAL':
            status = 'WAIT'; entry_class = 'WAIT'; reason = 'D1/H4 ยังไม่ให้ Macro direction'
        elif counter_macro:
            # Critical V4.7 rule: Long Hold must not produce an executable entry while
            # tactical H1/M15 is still opposite to D1/H4 macro.
            status = 'PRE-ENTRY' if (long_setup['ok'] or long_setup['near']) else 'WAIT'
            entry_class = 'PULLBACK / RESUME'
            reason = (f"Macro {direction} • Tactical {tactical} = pullback/rebound ภายใน Macro • "
                      f"รอ H1/M15 กลับ {direction} • {long_setup['name']} • {long_trig['name']}")
        elif long_setup['ok'] and long_trig['ok']:
            status = 'ENTRY READY'; entry_class = 'TREND ENTRY'
            reason = f"Macro/Tactical aligned • {long_setup['name']} • {long_trig['name']} • {long_loc['zone']}"
        elif long_setup['ok'] or long_setup['near']:
            status = 'PRE-ENTRY'; entry_class = 'TREND WATCH'
            reason = f"{long_setup['name']} • {long_trig['name']} • {long_loc['reason']}"
        else:
            status = 'WAIT'; entry_class = 'WAIT'; reason = long_setup['name']

    # Readiness describes setup quality, not probability of profit. Location is informative,
    # not a hard veto, so the engine does not become excessively restrictive.
    readiness = int(np.clip(
        0.30*ctx['macro_strength'] +
        0.25*ctx['tactical_strength'] +
        0.20*setup['score'] +
        0.15*trig['score'] +
        0.10*loc['score'], 0, 100))

    plan = make_plan(frames, direction, mode) if status == 'ENTRY READY' else None
    if status == 'ENTRY READY' and plan is None:
        status = 'PRE-ENTRY'
        entry_class = 'COUNTER-MACRO WATCH' if counter_macro else 'TREND WATCH'
        reason = 'สัญญาณครบ แต่โครงสร้าง SL กว้างเกินไป — รอราคาเข้าโครงสร้างใหม่'

    relation = 'ALIGNED' if aligned else 'COUNTER-MACRO' if counter_macro else 'MIXED'
    return {
        'status': status, 'direction': direction, 'macro': macro, 'tactical': tactical,
        'macro_strength': ctx['macro_strength'], 'tactical_strength': ctx['tactical_strength'],
        'relation': relation, 'entry_class': entry_class,
        'trend_d1': ctx['d1']['score'], 'h4_score': ctx['h4']['score'], 'h1_score': ctx['h1']['score'],
        'alignment': int(round((ctx['h4']['score']+ctx['h1']['score'])/2)),
        'setup': setup, 'trigger': trig, 'range': range_info(frames['15m'],48), 'location': loc,
        'readiness': readiness, 'plan': plan, 'reason': reason, 'states': ctx,
    }

def fmt(v):
    return '—' if v is None or not np.isfinite(v) else f'{v:,.2f}'


def render_card(result, title):
    ready = result['status'] == 'ENTRY READY'
    pre = result['status'] == 'PRE-ENTRY'
    counter = result['entry_class'] == 'COUNTER-MACRO ENTRY'
    icon = '🟢' if ready and result['direction'] == 'LONG' and not counter else '🔴' if ready and result['direction'] == 'SHORT' and not counter else '⚠️' if counter else '🟡'
    st.markdown(f'### {title}')
    st.markdown(f'**{icon} {result["status"]} — {result["direction"]}**')
    if counter:
        st.warning('COUNTER-MACRO ENTRY — สัญญาณนี้สวนทิศทาง D1/H4 และมีไว้สำหรับ Short Hold เท่านั้น')
    elif result['entry_class'] == 'PULLBACK / RESUME':
        st.info('PULLBACK / RESUME — Macro กับ Tactical ยังสวนกัน จึงยังไม่ถือเป็น Long Hold entry')
    elif result['entry_class'] == 'TREND ENTRY':
        st.caption('TREND ENTRY — Macro และ Tactical ไปทางเดียวกัน')
    a,b,c = st.columns(3)
    a.metric('Macro D1/H4', f"{result['macro']} {result['macro_strength']}/100")
    b.metric('Tactical H1/M15', f"{result['tactical']} {result['tactical_strength']}/100")
    c.metric('Setup Readiness', f"{result['readiness']}/100")
    st.caption(f"Signal Class: {result['entry_class']} • Macro {result['macro']} • Tactical {result['tactical']} • {result['relation']} • H4/H1 {result['location']['zone']} • M15 {result['setup']['name']} • M5 {result['trigger']['name']}")
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


st.title('Gold & Bitcoin Trading Analyzer — V4.7')
st.caption('D1 trend → H4/H1 structure & range → M15 setup → M5 entry trigger. Short Hold และ Long Hold แสดงพร้อมกัน')

with st.sidebar:
    st.header('ตั้งค่าการวิเคราะห์')
    asset_name = st.selectbox('สินทรัพย์', list(ASSETS.keys()), index=0)
    symbol = ASSETS[asset_name]
    chart_tf = st.selectbox('Timeframe กราฟ', list(TF.keys()), index=1)
    outputsize = st.slider('จำนวนแท่ง', 250, 800, 400, 50,
                            help='ค่ายิ่งมาก ยิ่งใช้ credit ต่อคำขอมากขึ้นบนแผนฟรี')
    auto = st.checkbox('Auto refresh', False)
    refresh = st.slider('รอบรีเฟรช (วินาที)', 60, 300, 90, 10,
                         help=f'ต่ำสุด 60s ให้สอดคล้องกับ cache ({CACHE_TTL}s) และ rate limit ของแผนฟรี')
    if st.button('รีเฟรชข้อมูลตอนนี้'):
        st.cache_data.clear(); st.rerun()
    st.divider()
    show_signals = st.checkbox('แสดงจุด BUY/SELL บนกราฟ (ย้อนหลัง)', True,
                                help='มาร์กเกอร์ดูสัญญาณเฉยๆ ไม่ได้ส่งคำสั่งจริง และไม่ได้กรองด้วย Macro/Tactical เหมือนการ์ดสถานะการเทรด')
    signal_threshold = st.slider('เกณฑ์คะแนนสัญญาณ (ยิ่งสูงยิ่งเข้ม)', 40, 90, 55, 5) if show_signals else 55
    st.divider()
    st.caption(f'API: Twelve Data • cache {CACHE_TTL}s • โหลดต่อครั้ง 5 คำขอ (ไม่มีคำขอซ้ำสำหรับกราฟ)')
    st.caption('ออกแบบให้ประหยัด API credit สำหรับ free-plan key: คำขอถูกหน่วงเวลา, cache ยาวขึ้น, และมี fallback ไปใช้ข้อมูลล่าสุดที่ดึงสำเร็จเมื่อคำขอถูก rate-limit')

if auto:
    st.markdown(f'<meta http-equiv="refresh" content="{refresh}">', unsafe_allow_html=True)

frames, frames_raw, errors, stale = {}, {}, {}, {}
for label in ['1D','4h','1h','15m','5m']:
    raw, err = get_ohlcv(symbol, TF[label], outputsize)
    if err:
        stale[label] = err if raw is not None and not raw.empty else None
        errors[label] = err
    frames_raw[label] = raw if raw is not None else pd.DataFrame()
    frames[label] = indicators(closed_only(frames_raw[label])) if not frames_raw[label].empty else pd.DataFrame()

# The chart timeframe is always one of the 5 already fetched above (TF keys match
# chart_tf's options), so it's reused here instead of firing a 6th API call — this
# also keeps the chart on the same closed-candle data the signals use, so the
# displayed "Entry" price and the chart/price header never disagree.
chart_raw = frames_raw.get(chart_tf, pd.DataFrame())
chart = frames.get(chart_tf, pd.DataFrame())

have_all = all(not frames[l].empty for l in ['1D','4h','1h','15m','5m'])
if errors:
    hard = {k: v for k, v in errors.items() if stale.get(k) is None}
    soft = {k: v for k, v in errors.items() if stale.get(k) is not None}
    if soft:
        st.warning('บางไทม์เฟรมใช้ข้อมูลล่าสุดที่แคชไว้ (อาจไม่ใหม่ที่สุด): ' + ' | '.join(f'{k}: {v}' for k, v in soft.items()))
    if hard:
        st.error(' | '.join(f'{k}: {v}' for k, v in hard.items()))
if chart.empty:
    st.error('ไม่มีข้อมูลสำหรับกราฟ/ราคาปัจจุบัน — อาจถูก rate-limit ลองกด "รีเฟรชข้อมูลตอนนี้" อีกครั้งในอีกสักครู่')
    st.stop()

# Use the freshest closed candle (5m) for the headline price whenever it's
# available, rather than whatever chart timeframe the user has selected. A
# 15m/1h/4h/1D "last close" can lag the actual market by up to that whole
# interval, which made the header disagree with Short Hold's 5m-based Entry
# by hundreds of dollars on a fast-moving asset like BTC. Falls back to the
# chart's own series if 5m data isn't available this refresh.
price_src = frames.get('5m') if not frames.get('5m', pd.DataFrame()).empty else chart
price = float(price_src.close.iloc[-1]); prev = float(price_src.close.iloc[-2]) if len(price_src)>1 else price
pct = (price/prev-1)*100 if prev else 0
st.subheader(asset_name); st.metric('Price', fmt(price), f'{pct:+.2f}%')

st.markdown('## สถานะการเทรด')
if have_all:
    short = evaluate(frames, 'Short Hold'); long = evaluate(frames, 'Long Hold')
    a,b = st.columns(2)
    with a: render_card(short, 'Short Hold')
    with b: render_card(long, 'Long Hold')
else:
    st.info('ข้อมูลบางไทม์เฟรมยังไม่พร้อม (rate-limit/ไม่มีข้อมูลแคชสำรอง) — รออีกสักครู่แล้วรีเฟรชใหม่')

st.markdown('## โครงสร้างตลาด')
rows=[]
for tf in ['1D','4h','1h','15m','5m']:
    d=frames[tf]
    if d.empty:
        rows.append({'TF':tf,'Bias':'—','Trend':'—','Structure':'—','Range':'—','Range High':'—','Range Low':'—'})
        continue
    s=tf_state(d); rg=range_info(d,48)
    rows.append({'TF':tf,'Bias':s['bias'],'Trend':s['score'],'Structure':s['structure'],'Range':rg['zone'],'Range High':rg['high'],'Range Low':rg['low']})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown(f'## กราฟ {chart_tf}')
fig=go.Figure(); fig.add_trace(go.Candlestick(x=chart.index,open=chart.open,high=chart.high,low=chart.low,close=chart.close,name='Price'))
for n in (20,50,200): fig.add_trace(go.Scatter(x=chart.index,y=chart[f'EMA{n}'],name=f'EMA{n}',mode='lines'))
if show_signals and not chart.empty:
    buys, sells = scan_signals(chart, signal_threshold)
    if buys:
        fig.add_trace(go.Scatter(
            x=[b['time'] for b in buys], y=[b['price'] for b in buys], mode='markers', name='BUY',
            marker=dict(symbol='triangle-up', size=11, color='#22c55e', line=dict(width=1, color='#ffffff')),
            text=[f"BUY {b['score']}%" for b in buys], hovertemplate='%{text}<br>%{y}<extra></extra>'))
    if sells:
        fig.add_trace(go.Scatter(
            x=[s['time'] for s in sells], y=[s['price'] for s in sells], mode='markers', name='SELL',
            marker=dict(symbol='triangle-down', size=11, color='#ef4444', line=dict(width=1, color='#ffffff')),
            text=[f"SELL {s['score']}%" for s in sells], hovertemplate='%{text}<br>%{y}<extra></extra>'))
fig.update_layout(height=520,xaxis_rangeslider_visible=False,margin=dict(l=10,r=10,t=30,b=10)); st.plotly_chart(fig,use_container_width=True)
if show_signals:
    st.caption('จุด BUY/SELL คือสัญญาณจากรูปแบบแท่งเทียนของไทม์เฟรมนี้เท่านั้น (ดูอย่างเดียว ไม่ส่งคำสั่งจริง) — ไม่ได้กรองด้วย Macro/Tactical เหมือนการ์ด "สถานะการเทรด" ด้านบน ดังนั้นอาจมีจุดที่สวนทางกับ Short/Long Hold ได้')

st.markdown('## หลักการของ V4.7')
st.write('D1/H4 = Macro. H1/M15 = Tactical. M15 = setup. M5 = trigger. Short Hold ตาม Tactical และสามารถเป็น COUNTER-MACRO ได้ แต่ระบบต้องติดป้ายชัดเจน. Long Hold ตาม Macro และห้ามแสดง ENTRY READY ขณะ Tactical ยังสวน Macro; ต้องรอ H1/M15 กลับทิศ. TREND ENTRY = Macro/Tactical aligned. COUNTER-MACRO ENTRY = Short Hold สวน Macro. PULLBACK / RESUME = Long Hold กำลังรอ Tactical กลับเข้า Macro. ระบบตรวจ setup/trigger ย้อนหลัง 3 แท่งที่ปิดแล้ว. Entry/SL/TP แสดงเฉพาะ ENTRY READY.')
