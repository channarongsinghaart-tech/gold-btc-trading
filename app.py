import os
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title='Gold & BTC Trading Analyzer V4.2.2', page_icon='📈', layout='wide')

BASE = 'https://api.twelvedata.com'
ASSETS = {'Gold XAU/USD': 'XAU/USD', 'Bitcoin BTC/USD': 'BTC/USD'}
TF = {'5m':'5min','15m':'15min','30m':'30min','1h':'1h','4h':'4h','1D':'1day'}
ANALYSIS_TFS = ['1D','4h','1h','30m']

API_KEY = os.getenv('TWELVEDATA_API_KEY','')
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
        return None, data.get('message','Twelve Data API error')
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
        return pd.DataFrame(), (data or {}).get('message','ไม่มีข้อมูล')
    df = pd.DataFrame(data['values'])
    for c in ['open','high','low','close','volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.sort_values('datetime').set_index('datetime')
    return df.dropna(subset=['open','high','low','close']), None


def add_indicators(df):
    d=df.copy(); c,h,l=d.close,d.high,d.low
    for n in (20,50,200): d[f'EMA{n}']=c.ewm(span=n,adjust=False).mean()
    delta=c.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    rs=gain/loss.replace(0,np.nan); d['RSI']=100-(100/(1+rs))
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean()
    d['MACD']=e12-e26; d['MACDsig']=d.MACD.ewm(span=9,adjust=False).mean(); d['MACDhist']=d.MACD-d.MACDsig
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    d['ATR']=tr.ewm(alpha=1/14,adjust=False).mean(); d['ATR_pct']=d.ATR/d.close*100
    if 'volume' in d.columns:
        d['VolMA20']=d.volume.rolling(20).mean(); d['VolRatio']=d.volume/d.VolMA20
    else: d['VolMA20']=np.nan; d['VolRatio']=np.nan
    d['body']=(d.close-d.open).abs(); d['range']=d.high-d.low
    d['body_ratio']=d.body/d.range.replace(0,np.nan)
    d['upper_wick']=d.high-d[['open','close']].max(axis=1)
    d['lower_wick']=d[['open','close']].min(axis=1)-d.low
    d['EMA20_slope']=d.EMA20.diff(5)
    return d


def structure_state(d, lookback=20):
    if len(d)<lookback+6: return 'MIXED'
    mid=max(4,lookback//3); recent=d.iloc[-mid:]; prior=d.iloc[-lookback:-mid]
    rh,rl=float(recent.high.max()),float(recent.low.min()); ph,pl=float(prior.high.max()),float(prior.low.min())
    if rh>ph and rl>pl: return 'HH_HL'
    if rh<ph and rl<pl: return 'LH_LL'
    x=d.iloc[-1]
    if x.EMA20>x.EMA50>x.EMA200: return 'BULL'
    if x.EMA20<x.EMA50<x.EMA200: return 'BEAR'
    return 'MIXED'


def bias(d):
    if d is None or len(d)<220: return 'NEUTRAL',50,'ข้อมูลไม่พอ'
    x=d.iloc[-1]; state=structure_state(d); bull=0; bear=0
    bull += int(x.close>x.EMA20); bear += int(x.close<x.EMA20)
    bull += int(x.EMA20>x.EMA50); bear += int(x.EMA20<x.EMA50)
    bull += int(x.EMA50>x.EMA200); bear += int(x.EMA50<x.EMA200)
    bull += int(x.MACD>x.MACDsig); bear += int(x.MACD<x.MACDsig)
    if state in ('HH_HL','BULL'): bull+=2
    elif state in ('LH_LL','BEAR'): bear+=2
    if bull>bear: return 'LONG',int(np.clip(50+(bull-bear)*12.5,0,100)),state
    if bear>bull: return 'SHORT',int(np.clip(50+(bear-bull)*12.5,0,100)),state
    return 'NEUTRAL',50,state


def levels(d, lookback=150):
    x=d.tail(lookback); ph=x.high.rolling(5,center=True).max(); pl=x.low.rolling(5,center=True).min()
    r=x.loc[x.high>=ph,'high'].dropna().tail(12); s=x.loc[x.low<=pl,'low'].dropna().tail(12)
    resistance=float(r.mean()) if len(r) else float(x.high.tail(30).max())
    support=float(s.mean()) if len(s) else float(x.low.tail(30).min())
    return support,resistance


def setup_flags(d):
    if len(d)<30: return {}
    x=d.iloc[-1]; prior=d.iloc[-21:-1]; atr=float(x.ATR)
    if not np.isfinite(atr) or atr<=0: return {}
    ph=float(prior.high.max()); pl=float(prior.low.min())
    bull_break=bool(x.close>ph and x.close>x.open and x.body_ratio>=.50 and x.body>=.35*atr)
    bear_break=bool(x.close<pl and x.close<x.open and x.body_ratio>=.50 and x.body>=.35*atr)
    bull_pull=bool(x.close>x.EMA20 and x.EMA20>x.EMA50 and x.low<=x.EMA20+.15*atr and x.close>x.open and x.body_ratio>=.35)
    bear_pull=bool(x.close<x.EMA20 and x.EMA20<x.EMA50 and x.high>=x.EMA20-.15*atr and x.close<x.open and x.body_ratio>=.35)
    recent_low=float(d.iloc[-8:-1].low.min()); recent_high=float(d.iloc[-8:-1].high.max())
    bull_sweep=bool(x.low<recent_low and x.close>recent_low and x.close>x.open)
    bear_sweep=bool(x.high>recent_high and x.close<recent_high and x.close<x.open)
    return {'bull_breakout':bull_break,'bear_breakdown':bear_break,'bull_pullback':bull_pull,'bear_pullback':bear_pull,'bull_sweep':bull_sweep,'bear_sweep':bear_sweep}


def fib_location(d, lookback=80):
    x=d.tail(lookback)
    hi=float(x.high.max()); lo=float(x.low.min()); span=hi-lo
    if span<=0: return {'high':hi,'low':lo,'zone_low':np.nan,'zone_high':np.nan,'price_pos':np.nan,'label':'N/A'}
    # For a bullish swing, discount is measured down from high; for bearish, premium is measured up from low.
    z705=hi-.705*span; z788=hi-.788*span; z886=hi-.886*span
    price=float(d.close.iloc[-1])
    pos=(price-lo)/span
    if z886 <= price <= z705: label='Bullish deep-discount zone'
    elif z705 < price <= hi: label='Premium / not ideal for long'
    elif lo <= price < z886: label='Below 0.886 — long invalidation zone'
    else: label='Neutral location'
    return {'high':hi,'low':lo,'z705':z705,'z788':z788,'z886':z886,'price_pos':pos,'label':label}


def volume_profile_proxy(d, lookback=120, bins=32):
    x=d.tail(lookback).copy()
    if x.empty: return {}
    lo=float(x.low.min()); hi=float(x.high.max()); span=hi-lo
    if span<=0: return {}
    typical=(x.high+x.low+x.close)/3
    vol=x.volume if 'volume' in x.columns else pd.Series(np.ones(len(x)),index=x.index)
    vol=pd.to_numeric(vol,errors='coerce').fillna(0)
    idx=np.clip(((typical-lo)/span*bins).astype(int),0,bins-1)
    profile=np.zeros(bins)
    for i,v in zip(idx,vol): profile[int(i)]+=float(v)
    if profile.max()<=0: return {}
    poc_idx=int(np.argmax(profile)); poc=lo+(poc_idx+.5)*span/bins
    # approximate value area = bins accumulating around POC until ~70% volume
    order=[poc_idx]; used=profile[poc_idx]; target=.70*profile.sum(); left=poc_idx-1; right=poc_idx+1
    while used<target and (left>=0 or right<bins):
        lv=profile[left] if left>=0 else -1; rv=profile[right] if right<bins else -1
        if rv>=lv: order.append(right); used+=max(rv,0); right+=1
        else: order.append(left); used+=max(lv,0); left-=1
    va_low=lo+(min(order)+.0)*span/bins; va_high=lo+(max(order)+1.0)*span/bins
    price=float(d.close.iloc[-1])
    if price<va_low: loc='Below value / discount'
    elif price>va_high: loc='Above value / premium'
    else: loc='Inside value'
    return {'poc':poc,'va_low':va_low,'va_high':va_high,'location':loc,'volume_total':float(vol.sum())}


def session_label(ts):
    h=ts.hour
    if 0<=h<7: return 'Asia'
    if 7<=h<12: return 'London'
    if 12<=h<17: return 'New York'
    return 'Late / overlap'


def proxy_orderflow(d):
    """Candle/volume proxy only. It is not true bid/ask delta or footprint data."""
    if d is None or len(d)<25: return {'score':50,'state':'NO DATA','reasons':[],'participation':np.nan}
    x=d.iloc[-1]; p=d.iloc[-2]; pp=d.iloc[-3]
    atr=float(x.ATR) if np.isfinite(x.ATR) else np.nan
    vr=float(x.VolRatio) if np.isfinite(x.VolRatio) else np.nan
    participation=vr
    score=50; reasons=[]
    rng=max(float(x.range),1e-12); body=float(x.body); upper=float(x.upper_wick); lower=float(x.lower_wick)
    high_part=np.isfinite(vr) and vr>=1.15
    low_part=np.isfinite(vr) and vr<0.75
    if high_part: reasons.append(f'participation {vr:.1f}x')
    if low_part: reasons.append(f'low participation {vr:.1f}x')

    bull_abs=(lower>=0.45*rng and x.close>x.open and high_part and (x.close-float(x.low))>=0.65*rng)
    bear_abs=(upper>=0.45*rng and x.close<x.open and high_part and (float(x.high)-x.close)>=0.65*rng)
    bull_flip=(p.close<p.open and x.close>x.open and x.close>p.high and body>=0.35*rng)
    bear_flip=(p.close>p.open and x.close<x.open and x.close<p.low and body>=0.35*rng)
    bull_progress=(x.close>p.close>pp.close)
    bear_progress=(x.close<p.close<pp.close)
    if bull_abs: score+=18; reasons.append('bullish absorption proxy')
    if bear_abs: score-=18; reasons.append('bearish absorption proxy')
    if bull_flip: score+=18; reasons.append('bullish dominance-shift proxy')
    if bear_flip: score-=18; reasons.append('bearish dominance-shift proxy')
    if bull_progress: score+=8
    if bear_progress: score-=8
    if low_part: score=50+(score-50)*0.55
    score=int(np.clip(round(score),0,100))
    if score>=68: state='BULLISH'
    elif score<=32: state='BEARISH'
    else: state='NEUTRAL'
    return {'score':score,'state':state,'reasons':reasons,'participation':participation,'bull_abs':bull_abs,'bear_abs':bear_abs,'bull_flip':bull_flip,'bear_flip':bear_flip}


def environment(d):
    if d is None or len(d)<80: return 'UNKNOWN',50,'ข้อมูลไม่พอ'
    x=d.iloc[-1]; atr_med=float(d.ATR.tail(50).median()); atr=float(x.ATR)
    slope=float(x.EMA20_slope); price=float(x.close)
    trend_strength=abs(float(x.EMA20-x.EMA50))/max(atr,1e-9)
    state=structure_state(d)
    compression=atr_med>0 and atr<0.75*atr_med
    expansion=atr_med>0 and atr>1.25*atr_med
    if compression: return 'COMPRESSION',int(np.clip(55+trend_strength*8,0,100)), 'ATR compressed vs 50-bar median'
    if state in ('HH_HL','BULL') and slope>0 and trend_strength>=0.35:
        return 'INITIATIVE UP',int(np.clip(60+trend_strength*12+(10 if expansion else 0),0,100)),'bullish structure + EMA slope'
    if state in ('LH_LL','BEAR') and slope<0 and trend_strength>=0.35:
        return 'INITIATIVE DOWN',int(np.clip(60+trend_strength*12+(10 if expansion else 0),0,100)),'bearish structure + EMA slope'
    return 'BALANCE',int(np.clip(55+trend_strength*8,0,100)),'mixed/rotational structure'


def v42_engine(frames, mode, flow):
    """Two trading horizons, both M30-primary.

    Short Hold: tighter M30 setup + stronger M5 proxy confirmation.
    Long Hold: wider M30 setup with H1/H4 regime; M5 is confirmatory, not mandatory.
    """
    details={}
    for tf in ANALYSIS_TFS:
        d=frames.get(tf)
        if d is not None and not d.empty:
            b,s,r=bias(d); env,envs,envr=environment(d)
            details[tf]={'Bias':b,'Strength':s,'Structure':structure_state(d),'Environment':env,'EnvScore':envs,'Reason':r}
    if any(tf not in details for tf in ANALYSIS_TFS):
        return 'NO TRADE',0,0,details,'MTF data incomplete'

    dirs={tf:details[tf]['Bias'] for tf in ANALYSIS_TFS}
    d1,h4,h1,m30=[dirs[x] for x in ANALYSIS_TFS]
    flow_dir='LONG' if flow['score']>=68 else 'SHORT' if flow['score']<=32 else 'NEUTRAL'

    # Score uses D1/H4/H1/M30, while entry horizon determines how much confirmation is required.
    weights={'1D':25,'4h':30,'1h':25,'30m':20}
    dominant = d1 if d1 in ('LONG','SHORT') else h4 if h4 in ('LONG','SHORT') else 'NO TRADE'
    if dominant=='NO TRADE':
        return 'NO TRADE',0,0,details,'No dominant higher-timeframe direction'

    aligned=sum(details[t]['Bias']==dominant for t in ANALYSIS_TFS)
    opposing=sum(details[t]['Bias'] not in (dominant,'NEUTRAL') for t in ANALYSIS_TFS)
    weighted=sum(weights[t]*details[t]['Strength']/100 for t in ANALYSIS_TFS if details[t]['Bias']==dominant)
    trend_score=int(np.clip(round(weighted/sum(weights.values())*100),0,100))
    align=int(round(100*sum(weights[t] for t in ANALYSIS_TFS if details[t]['Bias']==dominant)/sum(weights.values())))

    if mode=='1 — Short Hold':
        # Narrow signal: H4/H1/M30 must agree; D1 may be neutral but must not directly oppose.
        if h4!=dominant or h1!=dominant or m30!=dominant:
            return 'NO TRADE',trend_score,align,details,'Short Hold: H4/H1/M30 not fully aligned'
        if d1 not in (dominant,'NEUTRAL'):
            return 'NO TRADE',trend_score,align,details,'Short Hold: D1 conflicts'
        if trend_score<62:
            return 'NO TRADE',trend_score,align,details,'Short Hold: trend strength too weak'
        # M5 confirmation is mandatory and stronger than the old V4.2 gate.
        if flow_dir!=dominant or abs(flow['score']-50)<22:
            return 'NO TRADE',trend_score,align,details,'Short Hold: M5 confirmation not strong enough'
        final_score=int(np.clip(round(.62*trend_score+.18*align+.20*flow['score']),0,100))
        signal='SHORT HOLD LONG' if dominant=='LONG' else 'SHORT HOLD SHORT'
        return signal,final_score,align,details,'Short Hold: M30 setup + strong M5 confirmation'

    # Long Hold: D1/H4 define the regime. H1 or M30 can provide the setup confirmation.
    if h4!=dominant:
        return 'NO TRADE',trend_score,align,details,'Long Hold: H4 does not confirm regime'
    confirmations=sum(x==dominant for x in (h1,m30))
    if confirmations<1 or opposing>=2:
        return 'NO TRADE',trend_score,align,details,'Long Hold: insufficient H1/M30 confirmation'
    if trend_score<52:
        return 'NO TRADE',trend_score,align,details,'Long Hold: higher-timeframe trend too weak'
    final_score=int(np.clip(round(.70*trend_score+.20*align+.10*flow['score']),0,100))
    # M5 can weaken, but does not veto a longer-horizon setup.
    if flow_dir not in (dominant,'NEUTRAL'):
        final_score=max(0,final_score-8)
        note='M5 proxy opposes; long-hold setup remains valid only if M30 entry passes'
    else:
        note='M5 proxy supportive/neutral'
    signal='LONG HOLD LONG' if dominant=='LONG' else 'LONG HOLD SHORT'
    return signal,final_score,align,details,f'Long Hold: H1/M30 regime confirmation; {note}'


def entry_plan(frames, direction, support, resistance, mode, flow, loc, vp):
    if direction=='NO TRADE': return None,'No directional signal'
    base='LONG' if 'LONG' in direction else 'SHORT'
    m30=frames.get('30m'); h1=frames.get('1h'); h4=frames.get('4h')
    if any(x is None or x.empty for x in (m30,h1,h4)): return None,'MTF entry data incomplete'
    x=m30.iloc[-1]; atr=float(x.ATR); price=float(x.close)
    if not np.isfinite(atr) or atr<=0: return None,'ATR invalid'
    flags=setup_flags(m30)
    if base=='LONG':
        setup=flags.get('bull_pullback') or flags.get('bull_breakout') or flags.get('bull_sweep')
        if mode=='1 — Short Hold':
            near_ema=np.isfinite(x.EMA20) and abs(price-float(x.EMA20))<=.35*atr and price>float(x.EMA50)
            flow_ok=flow['score']>=72
            location_ok=(vp and vp.get('location') in ('Below value / discount','Inside value')) or loc.get('label','').startswith('Bullish deep')
            if not (setup or near_ema): return None,'Short Hold: รอ M30 pullback/breakout/sweep หรือ EMA20 ใกล้ ๆ'
            if not flow_ok: return None,'Short Hold: M5 confirmation ยังไม่แรงพอ'
            if not location_ok and not setup: return None,'Short Hold: location ยังไม่เหมาะ'
            if np.isfinite(resistance) and resistance>price and resistance-price<.80*atr: return None,'Short Hold: ใกล้ resistance เกินไป'
            invalidation=float(min(h1.tail(20).low.min(),h4.tail(12).low.min(),m30.tail(8).low.min()))
            sl=invalidation-.15*atr; risk=price-sl
            max_risk=2.0
            rr1,rr2=1.25,2.0
        else:
            near_ema=np.isfinite(x.EMA20) and abs(price-float(x.EMA20))<=.85*atr and price>float(x.EMA50)
            location_ok=(vp and vp.get('location') in ('Below value / discount','Inside value')) or loc.get('label','').startswith('Bullish deep')
            if not (setup or near_ema): return None,'Long Hold: รอ M30 pullback/breakout/sweep หรือ EMA20 proximity'
            if not location_ok and not setup: return None,'Long Hold: location ยังไม่เหมาะ'
            if np.isfinite(resistance) and resistance>price and resistance-price<.20*atr: return None,'Long Hold: ใกล้ resistance เกินไป'
            invalidation=float(min(h1.tail(40).low.min(),h4.tail(30).low.min(),m30.tail(16).low.min()))
            sl=invalidation-.30*atr; risk=price-sl
            max_risk=4.0
            rr1,rr2=1.5,3.0
        if risk<=0 or risk>max_risk*atr: return None,'Risk distance too large'
        tp1=price+rr1*risk; tp2=price+rr2*risk
        return {'direction':'LONG','entry':price,'sl':sl,'tp1':tp1,'tp2':tp2,'risk':risk,'rr1':rr1,'rr2':rr2,'invalidation':invalidation,'setup':'Pullback/Breakout/Sweep' if setup else 'EMA20 proximity','flow_score':flow['score'],'location':vp.get('location','') if vp else loc.get('label','')},'READY'

    setup=flags.get('bear_pullback') or flags.get('bear_breakdown') or flags.get('bear_sweep')
    if mode=='1 — Short Hold':
        near_ema=np.isfinite(x.EMA20) and abs(price-float(x.EMA20))<=.35*atr and price<float(x.EMA50)
        flow_ok=flow['score']<=28
        location_ok=(vp and vp.get('location') in ('Above value / premium','Inside value')) or loc.get('label','').startswith('Premium')
        if not (setup or near_ema): return None,'Short Hold: รอ M30 pullback/breakdown/sweep หรือ EMA20 ใกล้ ๆ'
        if not flow_ok: return None,'Short Hold: M5 confirmation ยังไม่แรงพอ'
        if not location_ok and not setup: return None,'Short Hold: location ยังไม่เหมาะ'
        if np.isfinite(support) and price>support and price-support<.80*atr: return None,'Short Hold: ใกล้ support เกินไป'
        invalidation=float(max(h1.tail(20).high.max(),h4.tail(12).high.max(),m30.tail(8).high.max()))
        sl=invalidation+.15*atr; risk=sl-price
        max_risk=2.0
        rr1,rr2=1.25,2.0
    else:
        near_ema=np.isfinite(x.EMA20) and abs(price-float(x.EMA20))<=.85*atr and price<float(x.EMA50)
        location_ok=(vp and vp.get('location') in ('Above value / premium','Inside value')) or loc.get('label','').startswith('Premium')
        if not (setup or near_ema): return None,'Long Hold: รอ M30 pullback/breakdown/sweep หรือ EMA20 proximity'
        if not location_ok and not setup: return None,'Long Hold: location ยังไม่เหมาะ'
        if np.isfinite(support) and price>support and price-support<.20*atr: return None,'Long Hold: ใกล้ support เกินไป'
        invalidation=float(max(h1.tail(40).high.max(),h4.tail(30).high.max(),m30.tail(16).high.max()))
        sl=invalidation+.30*atr; risk=sl-price
        max_risk=4.0
        rr1,rr2=1.5,3.0
    if risk<=0 or risk>max_risk*atr: return None,'Risk distance too large'
    tp1=price-rr1*risk; tp2=price-rr2*risk
    return {'direction':'SHORT','entry':price,'sl':sl,'tp1':tp1,'tp2':tp2,'risk':risk,'rr1':rr1,'rr2':rr2,'invalidation':invalidation,'setup':'Pullback/Breakdown/Sweep' if setup else 'EMA20 proximity','flow_score':flow['score'],'location':vp.get('location','') if vp else loc.get('label','')},'READY'





def regime_compatible(mode, frames, base):
    """Simple regime-aware quality check; it does not override the engine gate."""
    h4=frames.get('4h')
    h1=frames.get('1h')
    if h4 is None or h1 is None or h4.empty or h1.empty:
        return False
    e4=environment(h4)[0]; e1=environment(h1)[0]
    if base=='LONG':
        return e4 in ('INITIATIVE UP','BALANCE','COMPRESSION') and e1 in ('INITIATIVE UP','BALANCE','COMPRESSION')
    return e4 in ('INITIATIVE DOWN','BALANCE','COMPRESSION') and e1 in ('INITIATIVE DOWN','BALANCE','COMPRESSION')

def entry_quality(frames, direction, mode, flow, loc, vp, risk, vscore, alignment, details=None):
    """Separate setup quality from trend score. Returns 0-100 plus component breakdown.
    When there is NO TRADE, estimate readiness from the dominant higher-timeframe
    direction so the UI still shows why the setup is not ready.
    """
    details = details or {}
    if direction == 'NO TRADE':
        base = 'LONG' if details.get('1D',{}).get('Bias') == 'LONG' or (details.get('1D',{}).get('Bias') == 'NEUTRAL' and details.get('4h',{}).get('Bias') == 'LONG') else 'SHORT'
        if details.get('1D',{}).get('Bias') == 'SHORT' or (details.get('1D',{}).get('Bias') == 'NEUTRAL' and details.get('4h',{}).get('Bias') == 'SHORT'):
            base='SHORT'
    else:
        base='LONG' if 'LONG' in direction else 'SHORT'
    m30=frames.get('30m')
    if m30 is None or m30.empty:
        return 0, {}
    x=m30.iloc[-1]; atr=float(x.ATR); price=float(x.close)
    flags=setup_flags(m30)
    setup_keys=('bull_pullback','bull_breakout','bull_sweep') if base=='LONG' else ('bear_pullback','bear_breakdown','bear_sweep')
    setup_score=100 if any(flags.get(k,False) for k in setup_keys) else 55
    if np.isfinite(atr) and atr>0:
        ema_dist=abs(price-float(x.EMA20))/atr if np.isfinite(x.EMA20) else 9
        if mode=='1 — Short Hold': ema_score=int(np.clip(100-ema_dist*180,0,100))
        else: ema_score=int(np.clip(100-ema_dist*90,0,100))
    else: ema_score=0
    fs=int(flow.get('score',50))
    confirm=100-int(np.clip(abs(fs-(78 if base=='LONG' else 22))*2.0,0,100))
    if mode=='1 — Short Hold':
        confirm=int(np.clip(100-abs(fs-(82 if base=='LONG' else 18))*2.2,0,100))
    loc_text=(risk.get('location') or '')
    location=100 if (('discount' in loc_text.lower() or 'inside value' in loc_text.lower()) if base=='LONG' else ('premium' in loc_text.lower() or 'inside value' in loc_text.lower())) else 55
    if risk:
        rr=min(float(risk.get('rr2',0))/3.0*100,100)
    else:
        # Provisional R:R readiness: use the configured horizon's typical target multiple.
        rr_target = 2.0 if mode=='1 — Short Hold' else 3.0
        rr=min(rr_target/3.0*100,100)
    score=int(np.clip(round(.25*vscore+.15*alignment+.20*setup_score+.15*confirm+.10*location+.15*rr),0,100))
    return score, {'Trend':int(vscore),'Alignment':int(alignment),'Setup':int(setup_score),'Confirmation':int(confirm),'Location':int(location),'R:R':int(round(rr))}


def init_journal():
    if 'trade_journal' not in st.session_state:
        st.session_state.trade_journal=[]
    if 'active_trade_ids' not in st.session_state:
        st.session_state.active_trade_ids={}


def _signal_id(symbol, mode, risk, signal_bar):
    # Signal identity includes the completed M30 signal bar so a new setup is not
    # confused with an older setup in the same direction.
    ts=pd.Timestamp(signal_bar).strftime('%Y%m%d%H%M')
    return f"{symbol}|{mode}|{risk['direction']}|{ts}"


def _evaluate_trade_on_bars(trade, m30_bars):
    """Evaluate only bars strictly AFTER the signal bar.
    This prevents look-ahead from using the signal candle's pre-entry range."""
    if trade.get('Status') not in ('OPEN','TP1 HIT') or m30_bars is None or m30_bars.empty:
        return
    signal_bar=pd.Timestamp(trade['Signal Bar'])
    future=m30_bars[m30_bars.index > signal_bar]
    if future.empty:
        return

    for ts,row in future.iterrows():
        hi=float(row.high); lo=float(row.low)
        direction=trade['Direction']
        if direction=='LONG':
            hit_sl=lo<=trade['SL']; hit_tp1=hi>=trade['TP1']; hit_tp2=hi>=trade['TP2']
        else:
            hit_sl=hi>=trade['SL']; hit_tp1=lo<=trade['TP1']; hit_tp2=lo<=trade['TP2']

        # Without intrabar sequencing we cannot know which was hit first.
        # Review whenever SL and either target are both touched in the same bar.
        if hit_sl and (hit_tp1 or hit_tp2):
            trade['Status']='BOTH TOUCHED — REVIEW'
            trade['Result R']=np.nan
            trade['Exit Price']=np.nan
            trade['Resolved Bar']=pd.Timestamp(ts).strftime('%Y-%m-%d %H:%M UTC')
            return
        if hit_tp2:
            trade['Status']='TP2 HIT'
            trade['Result R']=trade['RR2']
            trade['Exit Price']=trade['TP2']
            trade['Resolved Bar']=pd.Timestamp(ts).strftime('%Y-%m-%d %H:%M UTC')
            return
        if hit_tp1 and trade.get('Status')=='OPEN':
            trade['Status']='TP1 HIT'
            trade['Result R']=np.nan
            trade['Exit Price']=trade['TP1']
            trade['TP1 Bar']=pd.Timestamp(ts).strftime('%Y-%m-%d %H:%M UTC')
            # Continue checking later bars for TP2/SL.
            continue
        if hit_sl:
            if trade.get('Status')=='TP1 HIT':
                trade['Status']='TP1 + SL'
                trade['Result R']=0.5*trade['RR1']-0.5
            else:
                trade['Status']='SL HIT'
                trade['Result R']=-1.0
            trade['Exit Price']=trade['SL']
            trade['Resolved Bar']=pd.Timestamp(ts).strftime('%Y-%m-%d %H:%M UTC')
            return


def update_journal(symbol, mode, direction, risk, signal_bar, m30_bars):
    init_journal()

    # Always advance existing trades, even when the current setup is NO TRADE.
    for trade in st.session_state.trade_journal:
        _evaluate_trade_on_bars(trade,m30_bars)

    if not risk or direction=='NO TRADE':
        return

    sid=_signal_id(symbol,mode,risk,signal_bar)
    existing=next((x for x in st.session_state.trade_journal if x['Signal ID']==sid),None)
    if existing is None:
        trade={
            'Signal ID':sid,
            'Signal Bar':pd.Timestamp(signal_bar).strftime('%Y-%m-%d %H:%M UTC'),
            'Time':pd.Timestamp(signal_bar).strftime('%Y-%m-%d %H:%M UTC'),
            'Symbol':symbol,
            'Mode':mode.split(' — ')[1] if ' — ' in mode else mode,
            'Direction':risk['direction'],
            'Entry':risk['entry'],'SL':risk['sl'],'TP1':risk['tp1'],'TP2':risk['tp2'],
            'Risk':risk['risk'],'RR1':risk['rr1'],'RR2':risk['rr2'],'Status':'OPEN',
            'Result R':np.nan,'Exit Price':np.nan,'TP1 Bar':'','Resolved Bar':''
        }
        st.session_state.trade_journal.append(trade)

def journal_stats():
    init_journal()
    df=pd.DataFrame(st.session_state.trade_journal)
    if df.empty: return df, {}
    closed=df[df['Result R'].notna()].copy()
    wins=closed[closed['Result R']>0]
    losses=closed[closed['Result R']<0]
    gross_profit=float(wins['Result R'].sum()) if not wins.empty else 0.0
    gross_loss=abs(float(losses['Result R'].sum())) if not losses.empty else 0.0
    pf=(gross_profit/gross_loss) if gross_loss>0 else np.nan
    eq=closed['Result R'].cumsum() if not closed.empty else pd.Series(dtype=float)
    dd=float((eq.cummax()-eq).max()) if not eq.empty else 0.0
    stats={'Trades':len(closed),'Open':int((df['Result R'].isna()).sum()),
           'Win Rate':(len(wins)/len(closed)*100 if len(closed) else np.nan),
           'Avg R':(float(closed['Result R'].mean()) if len(closed) else np.nan),
           'Profit Factor':pf,'Max DD R':dd}
    return df,stats

def fmt(v): return f'{v:,.4f}'

st.title('Gold & Bitcoin Trading Analyzer — V4.2.2')
st.caption('V4.2.2 = M30-primary + 2 horizons + Setup Readiness + look-ahead-safe journal. M5 is a proxy confirmation, not true footprint/delta/DOM.')

with st.sidebar:
    st.header('ตั้งค่าการวิเคราะห์')
    mode=st.radio('โหมดการถือ', ['1 — Short Hold','2 — Long Hold'], index=0, help='Short Hold = สัญญาณแคบและ M5 ต้องยืนยันแรง; Long Hold = ถือยาวกว่า ใช้ H1/H4 และ M30 เป็นหลัก')
    asset=st.selectbox('สินทรัพย์',list(ASSETS.keys()),index=0)
    timeframe=st.selectbox('Timeframe หลัก',list(TF.keys()),index=2)
    outputsize=st.select_slider('จำนวนแท่ง',options=[300,500,800],value=500)
    auto=st.checkbox('Auto refresh',value=False)
    refresh_seconds=st.slider('รอบรีเฟรช (วินาที)',30,300,60,step=30)
    if st.button('รีเฟรชข้อมูลตอนนี้',use_container_width=True):
        st.cache_data.clear(); st.rerun()
    st.divider(); st.write('API: Twelve Data')
    st.success('API key loaded') if API_KEY else st.error('ยังไม่มี API key')

symbol=ASSETS[asset]
if auto:
    time.sleep(refresh_seconds); st.rerun()

main_df,err=get_ohlcv(symbol,TF[timeframe],outputsize)
if err: st.error(err); st.stop()
if len(main_df)<220: st.warning('ข้อมูลน้อยกว่า 220 แท่ง — EMA200/MTF อาจยังไม่นิ่ง')

# Use only completed candles for analysis and signal generation. The newest API bar can still be forming.
main_full=add_indicators(main_df)
d=main_full.iloc[:-1].copy() if len(main_full)>1 else main_full.copy()
last_live=main_full.iloc[-1]
last=d.iloc[-1]
price=float(last_live.close); prev=float(main_full.close.iloc[-2]) if len(main_full)>=2 else price
pct=(price/prev-1)*100 if prev else 0.0
support,resistance=levels(d)

frames={}; errors={}
for tf in ANALYSIS_TFS:
    x,er=get_ohlcv(symbol,TF[tf],outputsize)
    if not er and len(x)>=220:
        xx=add_indicators(x)
        frames[tf]=xx.iloc[:-1].copy() if len(xx)>1 else xx.copy()
    else: errors[tf]=er or 'ข้อมูลไม่พอ'

# 30m is the primary setup timeframe; 5m is used only as the lowest-timeframe confirmation.
# Only completed 5m candles are used for confirmation.
five_df,five_err=get_ohlcv(symbol,'5min',max(250,min(800,outputsize)))
if not five_err and len(five_df)>=40:
    five_full=add_indicators(five_df)
    five=five_full.iloc[:-1].copy() if len(five_full)>1 else five_full.copy()
else:
    five=pd.DataFrame()
flow=proxy_orderflow(five)
loc=fib_location(five if not five.empty else d)
vp=volume_profile_proxy(five if not five.empty else d)

env,env_score,env_reason=environment(frames.get('4h',d))
vsignal,vscore,alignment,details,gate_reason=v42_engine(frames,mode,flow)
risk,entry_reason=entry_plan(frames,vsignal,support,resistance,mode,flow,loc,vp) if vsignal!='NO TRADE' else (None,'NO TRADE')
setup_readiness,quality_parts=entry_quality(frames,vsignal,mode,flow,loc,vp,risk,vscore,alignment,details)
regime=env

confidence=int(np.clip(.55*vscore+.25*alignment+.20*abs(flow['score']-50)*2,50,99)) if vsignal!='NO TRADE' else 0

c1,c2,c3,c4,c5,c6=st.columns(6)
c1.metric('Price',fmt(price),f'{pct:+.2f}%')
c2.metric('V4.2.2 Signal',vsignal)
c3.metric('Trend Score',f'{vscore}/100')
c4.metric('Setup Readiness',f'{setup_readiness}/100')
c5.metric('Confidence',f'{confidence}%' if confidence else '—')
c6.metric('RSI',f'{last.RSI:.1f}')
st.caption(f'แท่งล่าสุด: {last.name.strftime("%Y-%m-%d %H:%M UTC")} • {symbol} • {timeframe}')

fig=go.Figure(); fig.add_trace(go.Candlestick(x=d.index,open=d.open,high=d.high,low=d.low,close=d.close,name=symbol))
for col in ['EMA20','EMA50','EMA200']: fig.add_trace(go.Scatter(x=d.index,y=d[col],name=col,mode='lines'))
fig.add_hline(y=support,annotation_text='Support',line_dash='dot'); fig.add_hline(y=resistance,annotation_text='Resistance',line_dash='dot')
if vp:
    fig.add_hline(y=vp['poc'],annotation_text='Proxy POC',line_dash='dash')
fig.update_layout(height=620,xaxis_rangeslider_visible=False,margin=dict(l=10,r=10,t=30,b=10)); st.plotly_chart(fig,use_container_width=True)

left,right=st.columns(2)
with left:
    st.subheader('V4.2 Environment / Location')
    st.write(f'Environment: **{env}** ({env_score}/100)')
    st.write(f'Regime filter: **{"PASS" if (vsignal!="NO TRADE" and risk) else "WAIT"}**')
    st.write(f'Path/session: **{session_label(last.name)}**')
    st.write(f'Fib location: **{loc.get("label","N/A")}**')
    if 'z705' in loc: st.write(f'Fib 0.705 / 0.788 / 0.886: **{fmt(loc["z705"])} / {fmt(loc["z788"])} / {fmt(loc["z886"])}**')
    if vp: st.write(f'Proxy value: **{vp["location"]}** • POC **{fmt(vp["poc"])}**')
with right:
    st.subheader('Proxy Order Flow — 5m Confirmation')
    st.write(f'State: **{flow["state"]}** • Score **{flow["score"]}/100**')
    if np.isfinite(flow.get('participation',np.nan)): st.write(f'Participation: **{flow["participation"]:.2f}x** vs 20-bar average')
    st.write(' • '.join(flow['reasons']) if flow['reasons'] else 'ยังไม่พบ confirmation')
    st.caption('ใช้ candle + relative volume เป็น proxy เท่านั้น; ไม่ใช่ bid/ask delta, footprint หรือ DOM')

st.subheader('Entry / SL / TP')
if risk:
    st.success(f'ENTRY READY — {risk["direction"]}')
    st.write(f'Entry: **{fmt(risk["entry"])}**')
    st.write(f'Stop Loss: **{fmt(risk["sl"])}**')
    st.write(f'TP1: **{fmt(risk["tp1"])}** (R:R 1:{risk["rr1"]:.1f})')
    st.write(f'TP2: **{fmt(risk["tp2"])}** (R:R 1:{risk["rr2"]:.1f})')
    st.caption(f'Risk/Unit: {fmt(risk["risk"])} • Setup: {risk["setup"]} • Flow: {risk["flow_score"]}/100')
    st.caption(f'Signal ID: {_signal_id(symbol,mode,risk)}')
    st.write('Setup Readiness components: ' + ' • '.join(f'{k} {v}/100' for k,v in quality_parts.items()))
    st.caption('Setup Readiness คือความพร้อมของ setup ไม่ใช่ความน่าจะเป็นที่จะชนะ')
else:
    st.warning(f'WAIT / NO TRADE — {gate_reason}')
    st.write(f'Setup Readiness: **{setup_readiness}/100**')
    st.write('Setup Readiness components: ' + ' • '.join(f'{k} {v}/100' for k,v in quality_parts.items()))
    st.info(f'ต้องรอ: {gate_reason}')
    st.caption('Setup Readiness คือความพร้อมของ setup ไม่ใช่ความน่าจะเป็นที่จะชนะ และคำนวณได้แม้ยังไม่มี ENTRY READY')

# Journal uses the completed M30 signal bar and evaluates outcomes only on later M30 bars.
# This is deliberately look-ahead-safe. Existing trades are updated even when the current setup is NO TRADE.
update_journal(symbol,mode,vsignal,risk,last.name,d)

st.subheader('Trade Journal / Statistics')
jdf,jstats=journal_stats()
if jdf.empty:
    st.info('ยังไม่มีสัญญาณ ENTRY READY ที่ถูกบันทึกใน session นี้')
else:
    sc1,sc2,sc3,sc4,sc5,sc6=st.columns(6)
    sc1.metric('Closed Trades',jstats.get('Trades',0))
    sc2.metric('Win Rate',f"{jstats['Win Rate']:.1f}%" if np.isfinite(jstats.get('Win Rate',np.nan)) else '—')
    sc3.metric('Average R',f"{jstats['Avg R']:+.2f}" if np.isfinite(jstats.get('Avg R',np.nan)) else '—')
    sc4.metric('Profit Factor',f"{jstats['Profit Factor']:.2f}" if np.isfinite(jstats.get('Profit Factor',np.nan)) else '—')
    sc5.metric('Max DD',f"{jstats['Max DD R']:.2f}R")
    sc6.metric('Open',jstats.get('Open',0))
    show_cols=['Signal ID','Signal Bar','Symbol','Mode','Direction','Entry','SL','TP1','TP2','Status','Result R']
    st.dataframe(jdf[show_cols].tail(20),hide_index=True,use_container_width=True)
    csv=jdf.to_csv(index=False).encode('utf-8-sig')
    st.download_button('ดาวน์โหลด Trade Journal CSV',csv,file_name='v42_trade_journal.csv',mime='text/csv',use_container_width=True)
    if st.button('ล้าง Journal ใน Session',use_container_width=True):
        st.session_state.trade_journal=[]
        st.rerun()
    st.caption('Cooldown: 1 active setup ต่อ Symbol + Mode + Direction. Signal ID ผูกกับแท่ง M30 ที่ปิดแล้ว; ผลลัพธ์จะตรวจเฉพาะแท่ง M30 หลัง Entry. Journal อยู่ใน session นี้เท่านั้น')

st.subheader(f'Top-Down Multi-Timeframe — {mode}')
rows=[]
for tf in ANALYSIS_TFS:
    if tf in details:
        z=details[tf]; rows.append([tf,z['Bias'],z['Strength'],z['Structure'],z['Environment'],z['EnvScore'],z['Reason']])
    else: rows.append([tf,'ERROR','—','—','—','—',errors.get(tf,'ไม่มีข้อมูล')])
st.dataframe(pd.DataFrame(rows,columns=['TF','Bias','Strength','Structure','Environment','EnvScore','Reason']),hide_index=True,use_container_width=True)

st.subheader('Current Setup')
flags=setup_flags(d); active=[]
labels={'bull_breakout':'Bullish Breakout','bear_breakdown':'Bearish Breakdown','bull_pullback':'Bullish Pullback','bear_pullback':'Bearish Pullback','bull_sweep':'Bullish Liquidity Sweep','bear_sweep':'Bearish Liquidity Sweep'}
for k,v in flags.items():
    if v: active.append(labels[k])
st.write(' • '.join(active) if active else 'ยังไม่พบ setup ที่ผ่านโครงสร้างล่าสุด')

st.subheader('Indicator Snapshot')
st.dataframe(pd.DataFrame([
    ['EMA20',float(last.EMA20)],['EMA50',float(last.EMA50)],['EMA200',float(last.EMA200)],
    ['RSI14',float(last.RSI)],['MACD',float(last.MACD)],['MACD Signal',float(last.MACDsig)],['ATR14',float(last.ATR)],['ATR %',float(last.ATR_pct)]
],columns=['Indicator','Value']),hide_index=True,use_container_width=True)

st.caption('V4.2.2 ไม่ส่งคำสั่งซื้อขายอัตโนมัติ. M30 คือ timeframe หลัก; M5 เป็น order-flow proxy. Short Hold: สัญญาณแคบ/เข้าเร็วกว่า. Long Hold: โครงสร้างกว้าง/ถือยาวกว่า. Score/Confidence ไม่ใช่ win probability.')
