import os, time, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests

BASE = 'https://api.twelvedata.com'
ASSETS = {'Bitcoin BTC/USD': 'BTC/USD', 'Gold XAU/USD': 'XAU/USD'}
TF = {'5m': '5min', '15m': '15min', '1h': '1h', '4h': '4h', '1D': '1day'}

API_KEY = os.environ.get('TWELVEDATA_API_KEY', '').strip()
LINE_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN', '').strip()
LINE_USER_ID = os.environ.get('LINE_USER_ID', '').strip()

MIN_CALL_INTERVAL = 1.2
OUTPUTSIZE = 400
_last_call = 0.0


def api_get(symbol, interval):
    global _last_call
    wait = MIN_CALL_INTERVAL - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    r = requests.get(
        f'{BASE}/time_series',
        params={
            'symbol': symbol, 'interval': interval, 'outputsize': OUTPUTSIZE,
            'apikey': API_KEY, 'timezone': 'UTC'
        }, timeout=25)
    _last_call = time.time()
    try:
        data = r.json()
    except Exception:
        return pd.DataFrame(), f'HTTP {r.status_code}: invalid JSON'
    if r.status_code != 200 or ('code' in data and int(data.get('code', 0)) >= 400):
        return pd.DataFrame(), data.get('message', f'Twelve Data HTTP {r.status_code}')
    if 'values' not in data:
        return pd.DataFrame(), data.get('message', 'ไม่มีข้อมูล')
    df = pd.DataFrame(data['values'])
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.sort_values('datetime').set_index('datetime')
    return df.dropna(subset=['open','high','low','close']), None


def indicators(df):
    d = df.copy()
    if d.empty:
        return d
    c, h, l = d.close, d.high, d.low
    for n in (20, 50, 200):
        d[f'EMA{n}'] = c.ewm(span=n, adjust=False).mean()
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    d['ATR'] = tr.ewm(alpha=1/14, adjust=False).mean()
    d['range'] = h-l
    d['body'] = (c-d.open).abs()
    d['body_ratio'] = d.body / d['range'].replace(0, np.nan)
    d['upper_wick'] = h-d[['open','close']].max(axis=1)
    d['lower_wick'] = d[['open','close']].min(axis=1)-l
    if 'volume' in d.columns:
        d['VolMA20'] = d.volume.rolling(20).mean()
        d['VolRatio'] = d.volume / d.VolMA20.replace(0, np.nan)
    else:
        d['VolMA20'] = np.nan; d['VolRatio'] = np.nan
    return d


def closed_only(d):
    # Twelve Data normally returns the currently forming candle as the newest row.
    return d.iloc[:-1].copy() if d is not None and len(d) >= 3 else d


def structure(d, lookback=20):
    if d is None or len(d) < lookback + 8:
        return 'MIXED'
    recent = d.iloc[-6:]; prior = d.iloc[-lookback:-6]
    rh, rl = float(recent.high.max()), float(recent.low.min())
    ph, pl = float(prior.high.max()), float(prior.low.min())
    if rh > ph and rl > pl: return 'HH_HL'
    if rh < ph and rl < pl: return 'LH_LL'
    x = d.iloc[-1]
    if x.EMA20 > x.EMA50 > x.EMA200: return 'BULL'
    if x.EMA20 < x.EMA50 < x.EMA200: return 'BEAR'
    return 'MIXED'


def tf_state(d):
    if d is None or len(d) < 205:
        return {'bias':'NEUTRAL','score':50,'structure':'MIXED'}
    x = d.iloc[-1]
    stc = structure(d)
    ema_bull = x.EMA20 > x.EMA50 > x.EMA200
    ema_bear = x.EMA20 < x.EMA50 < x.EMA200
    slope = float(x.EMA20 - d.EMA20.iloc[-6])
    bull = int(ema_bull) + int(x.close > x.EMA20) + int(slope > 0) + 2*int(stc in ('HH_HL','BULL'))
    bear = int(ema_bear) + int(x.close < x.EMA20) + int(slope < 0) + 2*int(stc in ('LH_LL','BEAR'))
    diff = bull-bear
    bias = 'LONG' if diff > 0 else 'SHORT' if diff < 0 else 'NEUTRAL'
    score = int(np.clip(50 + abs(diff)*10, 0, 100))
    return {'bias':bias,'score':score,'structure':stc}


def range_info(d, n=48):
    x=d.tail(min(n,len(d))); hi=float(x.high.max()); lo=float(x.low.min())
    span=max(hi-lo,1e-9); price=float(d.close.iloc[-1]); pos=(price-lo)/span
    zone='LOWER RANGE' if pos<0.33 else 'UPPER RANGE' if pos>0.67 else 'MID RANGE'
    return {'zone':zone,'high':hi,'low':lo,'pos':pos}


def macro_context(d1,h4,h1,m15):
    s1,s4,sH,s15=[tf_state(x) for x in (d1,h4,h1,m15)]
    if s1['bias']==s4['bias'] and s1['bias'] in ('LONG','SHORT'):
        macro=s1['bias']; macro_strength=int(round((s1['score']+s4['score'])/2))
    else:
        macro='NEUTRAL'; macro_strength=int(round((s1['score']+s4['score'])/2))
    if sH['bias']==s15['bias'] and sH['bias'] in ('LONG','SHORT'):
        tactical=sH['bias']; tactical_strength=int(round((sH['score']+s15['score'])/2))
    else:
        tactical='NEUTRAL'; tactical_strength=int(round((sH['score']+s15['score'])/2))
    return {'d1':s1,'h4':s4,'h1':sH,'m15':s15,'macro':macro,'macro_strength':macro_strength,
            'tactical':tactical,'tactical_strength':tactical_strength}


def m15_candidate(row, d, direction):
    atr=max(float(row.ATR),1e-9); rg=max(float(row.range),1e-9); ema20=float(row.EMA20); ema50=float(row.EMA50)
    hi20=float(d.iloc[-21:-1].high.max()); lo20=float(d.iloc[-21:-1].low.min())
    recent_low=float(d.iloc[-8:-1].low.min()); recent_high=float(d.iloc[-8:-1].high.max())
    bull_pull=row.low <= ema20+0.75*atr and row.close>=ema20 and row.close>row.open and row.close>=row.low+0.50*rg
    bear_pull=row.high >= ema20-0.75*atr and row.close<=ema20 and row.close<row.open and row.close<=row.high-0.50*rg
    bull_sweep=row.low<recent_low and row.close>recent_low and row.close>row.open
    bear_sweep=row.high>recent_high and row.close<recent_high and row.close<row.open
    bull_break=row.close>hi20 and row.close>row.open and row.body_ratio>=0.30
    bear_break=row.close<lo20 and row.close<row.open and row.body_ratio>=0.30
    bull_cont=row.close>ema20 and ema20>=ema50*0.998 and row.low>float(d.iloc[-4:-1].low.min())
    bear_cont=row.close<ema20 and ema20<=ema50*1.002 and row.high<float(d.iloc[-4:-1].high.max())
    names=[]
    if direction=='LONG':
        if bull_pull: names.append('M15 pullback')
        if bull_sweep: names.append('M15 sweep-reclaim')
        if bull_break: names.append('M15 breakout')
        if bull_cont: names.append('M15 continuation')
    else:
        if bear_pull: names.append('M15 pullback')
        if bear_sweep: names.append('M15 sweep-reject')
        if bear_break: names.append('M15 breakdown')
        if bear_cont: names.append('M15 continuation')
    return names


def m15_setup(d, direction):
    if d is None or len(d)<80 or direction not in ('LONG','SHORT'):
        return {'ok':False,'score':0,'name':'รอ M15 setup','age':None}
    candidates=[]
    for off in (0,1,2):
        sub=d if off==0 else d.iloc[:-off]
        if len(sub)<30: continue
        for name in m15_candidate(sub.iloc[-1],sub,direction):
            candidates.append((off,name))
    if candidates:
        newest=min(x[0] for x in candidates); names=[]
        for off,name in candidates:
            if off==newest and name not in names: names.append(name)
        return {'ok':True,'score':min(100,62+8*len(names)),'name':' + '.join(names),'age':newest}
    return {'ok':False,'score':35,'name':'รอ M15 pullback / breakout / sweep / continuation','age':None}


def m5_trigger(d, direction):
    if d is None or len(d)<50 or direction not in ('LONG','SHORT'):
        return {'ok':False,'score':0,'name':'รอ M5 trigger','fresh':False,'age':None}
    x,p=d.iloc[-1],d.iloc[-2]; rg=max(float(x.range),1e-9)
    bull_break=x.close>p.high and x.close>x.open and x.body_ratio>=0.20
    bear_break=x.close<p.low and x.close<x.open and x.body_ratio>=0.20
    bull_reclaim=x.close>x.EMA20 and p.close<=p.EMA20 and x.close>x.open
    bear_reclaim=x.close<x.EMA20 and p.close>=p.EMA20 and x.close<x.open
    bull_reject=x.lower_wick>=0.20*rg and x.close>x.open and x.close>=x.low+0.55*rg
    bear_reject=x.upper_wick>=0.20*rg and x.close<x.open and x.close<=x.high-0.55*rg
    if direction=='LONG':
        event_score=40*int(bull_break)+35*int(bull_reclaim)+25*int(bull_reject)
        name='M5 bullish trigger'
    else:
        event_score=40*int(bear_break)+35*int(bear_reclaim)+25*int(bear_reject)
        name='M5 bearish trigger'
    ok=bool(event_score>=55)
    return {'ok':ok,'score':int(event_score),'name':name if ok else ('รอ M5 กลับขึ้น' if direction=='LONG' else 'รอ M5 กลับลง'),'fresh':ok,'age':0 if ok else None}


def location(h4,h1,direction):
    if direction not in ('LONG','SHORT'): return {'score':50,'zone':'MIXED'}
    zones=[range_info(h4)['zone'],range_info(h1)['zone']]
    if direction=='LONG':
        if zones.count('LOWER RANGE')==2: return {'score':90,'zone':'LOWER RANGE'}
        if 'LOWER RANGE' in zones: return {'score':75,'zone':'LOWER/MID'}
        if zones.count('UPPER RANGE')==2: return {'score':30,'zone':'UPPER RANGE'}
    else:
        if zones.count('UPPER RANGE')==2: return {'score':90,'zone':'UPPER RANGE'}
        if 'UPPER RANGE' in zones: return {'score':75,'zone':'UPPER/MID'}
        if zones.count('LOWER RANGE')==2: return {'score':30,'zone':'LOWER RANGE'}
    return {'score':55,'zone':'MID RANGE'}


def make_plan(frames,direction,mode):
    m5,m15,h1=frames['5m'],frames['15m'],frames['1h']
    if direction not in ('LONG','SHORT'): return None
    if mode=='Short Hold':
        entry=float(m5.close.iloc[-1]); atr=max(float(m5.ATR.iloc[-1]),1e-9)
        if direction=='LONG':
            sl=float(m5.tail(12).low.min())-0.25*atr; risk=entry-sl
            if risk<1.0*atr or risk>2.8*atr: return None
            tp1,tp2=entry+1.2*risk,entry+1.8*risk
        else:
            sl=float(m5.tail(12).high.max())+0.25*atr; risk=sl-entry
            if risk<1.0*atr or risk>2.8*atr: return None
            tp1,tp2=entry-1.2*risk,entry-1.8*risk
    else:
        entry=float(m15.close.iloc[-1]); atr=max(float(m15.ATR.iloc[-1]),1e-9)
        if direction=='LONG':
            sl=min(float(m15.tail(14).low.min()),float(h1.tail(10).low.min()))-0.30*atr; risk=entry-sl
            if risk<1.2*atr or risk>5.0*atr: return None
            tp1,tp2=entry+1.5*risk,entry+3.0*risk
        else:
            sl=max(float(m15.tail(14).high.max()),float(h1.tail(10).high.max()))+0.30*atr; risk=sl-entry
            if risk<1.2*atr or risk>5.0*atr: return None
            tp1,tp2=entry-1.5*risk,entry-3.0*risk
    return {'entry':entry,'sl':sl,'tp1':tp1,'tp2':tp2,'risk':risk,'rr':abs(tp2-entry)/risk}


def evaluate(frames, mode):
    ctx=macro_context(frames['1D'],frames['4h'],frames['1h'],frames['15m'])
    # Trend-following only: all four decision timeframes must agree.
    dirs=[ctx['d1']['bias'],ctx['h4']['bias'],ctx['h1']['bias'],ctx['m15']['bias']]
    aligned=dirs[0] in ('LONG','SHORT') and all(x==dirs[0] for x in dirs)
    direction=dirs[0] if aligned else ('LONG' if ctx['tactical']=='LONG' else 'SHORT' if ctx['tactical']=='SHORT' else 'NEUTRAL')
    macro=ctx['macro']; tactical=ctx['tactical']
    setup=m15_setup(frames['15m'],direction)
    trig=m5_trigger(frames['5m'],direction)
    loc=location(frames['4h'],frames['1h'],direction)
    macro_strength=ctx['macro_strength']
    if aligned and macro_strength>=60 and setup['ok'] and trig['ok'] and trig['fresh']:
        status='ENTRY READY'; entry_class='TREND ENTRY'
        reason=f'D1/H4/H1/M15 align {direction} • {setup["name"]} • {trig["name"]} • M5 {trig["score"]}/100'
    elif not aligned:
        status='WAIT'; entry_class='WAIT'; reason='D1/H4/H1/M15 ยังไม่ align — ไม่สวน Macro'
    elif macro_strength<60:
        status='WAIT'; entry_class='WAIT'; reason=f'Macro strength {macro_strength}/100 ต่ำกว่า 60'
    elif not setup['ok']:
        status='PRE-ENTRY'; entry_class='TREND WATCH'; reason=setup['name']
    elif not trig['ok']:
        status='PRE-ENTRY'; entry_class='TREND WATCH'; reason=f'{setup["name"]} • {trig["name"]}'
    else:
        status='PRE-ENTRY'; entry_class='TREND WATCH'; reason='รอเงื่อนไข ENTRY READY เพิ่มเติม'
    plan=make_plan(frames,direction,mode) if status=='ENTRY READY' else None
    if status=='ENTRY READY' and plan is None:
        status='PRE-ENTRY'; entry_class='TREND WATCH'; reason='สัญญาณครบ แต่โครงสร้าง SL อยู่นอก risk gate'
    readiness=int(np.clip(0.30*macro_strength+0.25*ctx['tactical_strength']+0.20*setup['score']+0.15*trig['score']+0.10*loc['score'],0,100))
    return {'status':status,'direction':direction,'macro':macro,'tactical':tactical,'macro_strength':macro_strength,
            'tactical_strength':ctx['tactical_strength'],'setup':setup,'trigger':trig,'location':loc,'readiness':readiness,
            'plan':plan,'reason':reason,'states':ctx}


def push_line(text):
    if not LINE_TOKEN or not LINE_USER_ID:
        return False, 'LINE secrets missing'
    r=requests.post('https://api.line.me/v2/bot/message/push',
        headers={'Authorization':f'Bearer {LINE_TOKEN}','Content-Type':'application/json'},
        json={'to':LINE_USER_ID,'messages':[{'type':'text','text':text[:4900]}]},timeout=20)
    if r.status_code==200: return True,'OK'
    try: detail=r.json().get('message',r.text)
    except Exception: detail=r.text
    return False,f'HTTP {r.status_code}: {detail}'


def format_alert(asset,mode,result,checked_at):
    p=result['plan']; d=result['direction']
    return (f'🚨 ENTRY READY — {asset}\n'
            f'Mode: {mode}\nDirection: {d}\n'
            f'Entry: {p["entry"]:,.2f}\nSL: {p["sl"]:,.2f}\n'
            f'TP1: {p["tp1"]:,.2f}\nTP2: {p["tp2"]:,.2f}\n'
            f'R:R TP2: 1:{p["rr"]:.2f}\n'
            f'Macro: {result["macro"]} {result["macro_strength"]}/100\n'
            f'Tactical: {result["tactical"]} {result["tactical_strength"]}/100\n'
            f'M15: {result["setup"]["name"]}\nM5: {result["trigger"]["name"]} {result["trigger"]["score"]}/100\n'
            f'Checked: {checked_at}')


def main():
    if not API_KEY:
        raise SystemExit('Missing TWELVEDATA_API_KEY')
    if not LINE_TOKEN or not LINE_USER_ID:
        raise SystemExit('Missing LINE_CHANNEL_ACCESS_TOKEN or LINE_USER_ID')
    now=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    alerts=[]
    for asset,symbol in ASSETS.items():
        frames={}; errors=[]
        for label in ['1D','4h','1h','15m','5m']:
            raw,err=api_get(symbol,TF[label])
            if err: errors.append(f'{label}: {err}')
            else: frames[label]=indicators(closed_only(raw))
        if errors:
            print(f'[{asset}] API errors: ' + ' | '.join(errors)); continue
        for mode in ['Short Hold','Long Hold']:
            result=evaluate(frames,mode)
            print(json.dumps({'asset':asset,'mode':mode,'status':result['status'],'direction':result['direction'],
                              'macro':result['macro'],'macro_strength':result['macro_strength'],
                              'tactical':result['tactical'],'readiness':result['readiness']},ensure_ascii=False))
            if result['status']=='ENTRY READY' and result['plan']:
                alerts.append(format_alert(asset,mode,result,now))
    if alerts:
        # One LINE message per qualifying setup. The workflow runs on the closed M5 candle,
        # so only fresh triggers can generate an alert.
        for msg in alerts:
            ok,detail=push_line(msg); print('LINE:',detail)
            if not ok: raise SystemExit(detail)
    else:
        print('No ENTRY READY setups.')

if __name__=='__main__':
    main()
