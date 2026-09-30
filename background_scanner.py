import os, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import requests

BASE='https://api.twelvedata.com'
ASSETS={'Bitcoin BTC/USD':'BTC/USD','Gold XAU/USD':'XAU/USD'}
TF={'1D':'1day','4h':'4h','1h':'1h','15m':'15min','5m':'5min'}
TOKEN=os.getenv('LINE_CHANNEL_ACCESS_TOKEN','').strip()
USER_ID=os.getenv('LINE_USER_ID','').strip()
API_KEY=os.getenv('TWELVEDATA_API_KEY','').strip()
STATE=Path('alert_state.json')

# Free-plan Twelve Data key: 8 credits/minute, shared across BOTH assets in
# this run AND anything else using the same key (e.g. the Streamlit app, if
# open at the same time). Track every call's timestamp in a rolling 60s
# window and wait before any call that would push the count over the cap,
# with margin to spare (cap at 6, not 8, since timing/clock jitter on a
# GitHub Actions runner can shift a call across the window boundary).
_CALL_TIMES=[]
MAX_CALLS_PER_MIN=6

def _pace():
    now=time.monotonic()
    while _CALL_TIMES and now-_CALL_TIMES[0]>60:
        _CALL_TIMES.pop(0)
    if len(_CALL_TIMES)>=MAX_CALLS_PER_MIN:
        wait=60-(now-_CALL_TIMES[0])+0.5
        if wait>0:time.sleep(wait)
        now=time.monotonic()
        while _CALL_TIMES and now-_CALL_TIMES[0]>60:
            _CALL_TIMES.pop(0)
    _CALL_TIMES.append(time.monotonic())

def fetch(symbol, interval, outputsize=260, _retried=False):
    _pace()
    r=requests.get(f'{BASE}/time_series',params={'symbol':symbol,'interval':interval,'outputsize':outputsize,'apikey':API_KEY,'timezone':'UTC'},timeout=20)
    data=r.json()
    if r.status_code!=200 or 'values' not in data:
        msg=data.get('message',f'Twelve Data HTTP {r.status_code}')
        rate_limited=r.status_code==429 or 'run out of api credits' in msg.lower() or 'credits were used' in msg.lower()
        if rate_limited and not _retried:
            print(f'[RATE LIMIT] {symbol} {interval}: {msg} -- waiting 65s and retrying once')
            time.sleep(65)
            return fetch(symbol, interval, outputsize, _retried=True)
        raise RuntimeError(msg)
    d=pd.DataFrame(data['values'])
    for c in ['open','high','low','close','volume']:
        if c in d: d[c]=pd.to_numeric(d[c],errors='coerce')
    d['datetime']=pd.to_datetime(d['datetime'],utc=True); d=d.sort_values('datetime').set_index('datetime')
    d=d.dropna(subset=['open','high','low','close'])
    return d.iloc[:-1].copy() if len(d)>3 else d

def ind(d):
    d=d.copy(); c,h,l=d.close,d.high,d.low
    for n in (20,50,200): d[f'EMA{n}']=c.ewm(span=n,adjust=False).mean()
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    d['ATR']=tr.ewm(alpha=1/14,adjust=False).mean(); d['range']=h-l; d['body']=(c-d.open).abs(); d['body_ratio']=d.body/d['range'].replace(0,np.nan)
    d['upper_wick']=h-d[['open','close']].max(axis=1); d['lower_wick']=d[['open','close']].min(axis=1)
    d['VolMA20']=d.volume.rolling(20).mean() if 'volume' in d else np.nan; d['VolRatio']=d.volume/d.VolMA20.replace(0,np.nan) if 'volume' in d else np.nan
    return d

def state(d):
    if len(d)<205:return 'NEUTRAL',50
    x=d.iloc[-1]; recent=d.iloc[-6:]; prior=d.iloc[-20:-6]; rh,rl=recent.high.max(),recent.low.min(); ph,pl=prior.high.max(),prior.low.min()
    st='HH_HL' if rh>ph and rl>pl else 'LH_LL' if rh<ph and rl<pl else 'BULL' if x.EMA20>x.EMA50>x.EMA200 else 'BEAR' if x.EMA20<x.EMA50<x.EMA200 else 'MIXED'
    bull=int(x.EMA20>x.EMA50>x.EMA200)+int(x.close>x.EMA20)+int(x.EMA20>d.EMA20.iloc[-6])+2*int(st in ('HH_HL','BULL'))
    bear=int(x.EMA20<x.EMA50<x.EMA200)+int(x.close<x.EMA20)+int(x.EMA20<d.EMA20.iloc[-6])+2*int(st in ('LH_LL','BEAR'))
    return ('LONG' if bull>bear else 'SHORT' if bear>bull else 'NEUTRAL'), int(np.clip(50+abs(bull-bear)*10,0,100))

def setup(d,direction):
    if len(d)<80 or direction=='NEUTRAL': return False,0,'wait'
    x=d.iloc[-1]; p=d.iloc[-2]; atr=max(float(x.ATR),1e-9); hi=float(d.iloc[-21:-1].high.max()); lo=float(d.iloc[-21:-1].low.min())
    if direction=='LONG':
        pull=x.close>x.EMA20 and x.EMA20>x.EMA50 and x.low<=x.EMA20+0.75*atr and x.close>x.open and x.body_ratio>=0.30
        sweep=x.low<d.iloc[-8:-1].low.min() and x.close>d.iloc[-8:-1].low.min() and x.close>x.open
        br=x.close>hi and x.close>x.open and x.body_ratio>=0.30
        ok=pull or sweep or br; return bool(ok), min(100,62+8*sum([pull,sweep,br])), 'M15 pullback/sweep/breakout' if ok else 'wait'
    pull=x.close<x.EMA20 and x.EMA20<x.EMA50 and x.high>=x.EMA20-0.75*atr and x.close<x.open and x.body_ratio>=0.30
    sweep=x.high>d.iloc[-8:-1].high.max() and x.close<d.iloc[-8:-1].high.max() and x.close<x.open
    br=x.close<lo and x.close<x.open and x.body_ratio>=0.30
    ok=pull or sweep or br; return bool(ok), min(100,62+8*sum([pull,sweep,br])), 'M15 pullback/sweep/breakdown' if ok else 'wait'

def trigger(d,direction):
    if len(d)<50 or direction=='NEUTRAL': return False,0
    x,p=d.iloc[-1],d.iloc[-2]; rg=max(float(x.range),1e-9)
    if direction=='LONG':
        br=x.close>p.high and x.close>x.open and x.body_ratio>=0.20; rec=x.close>x.EMA20 and p.close<=p.EMA20 and x.close>x.open; rej=x.lower_wick>=0.20*rg and x.close>x.open and x.close>=x.low+0.55*rg
    else:
        br=x.close<p.low and x.close<x.open and x.body_ratio>=0.20; rec=x.close<x.EMA20 and p.close>=p.EMA20 and x.close<x.open; rej=x.upper_wick>=0.20*rg and x.close<x.open and x.close<=x.high-0.55*rg
    vol=float(x.VolRatio) if np.isfinite(x.VolRatio) else 1.0; side=(x.close>x.EMA20) if direction=='LONG' else (x.close<x.EMA20)
    score=min(100,40*int(br)+35*int(rec)+25*int(rej)+10*int(vol>=1.0)+20*int(side)); ok=(br or rec or rej) and score>=55
    return bool(ok),int(score)

def plan(fr,direction,mode):
    m5,m15,h1=fr['5m'],fr['15m'],fr['1h']; entry=float(m5.close.iloc[-1] if mode=='Short Hold' else m15.close.iloc[-1]); atr=float(m5.ATR.iloc[-1] if mode=='Short Hold' else m15.ATR.iloc[-1])
    if mode=='Short Hold':
        sl=(float(m5.tail(12).low.min())-0.25*atr) if direction=='LONG' else (float(m5.tail(12).high.max())+0.25*atr); maxr=2.8
        r=entry-sl if direction=='LONG' else sl-entry
        if r<=0 or r>maxr*atr:return None
        tp1,tp2=(entry+1.2*r,entry+1.8*r) if direction=='LONG' else (entry-1.2*r,entry-1.8*r)
    else:
        sl=(min(float(m15.tail(14).low.min()),float(h1.tail(10).low.min()))-0.30*atr) if direction=='LONG' else (max(float(m15.tail(14).high.max()),float(h1.tail(10).high.max()))+0.30*atr); maxr=5.0
        r=entry-sl if direction=='LONG' else sl-entry
        if r<=0 or r>maxr*atr:return None
        tp1,tp2=(entry+1.5*r,entry+3*r) if direction=='LONG' else (entry-1.5*r,entry-3*r)
    return entry,sl,tp1,tp2

def load_state():
    try:return json.loads(STATE.read_text())
    except:return {}

def save_state(s): STATE.write_text(json.dumps(s,indent=2))

def push(text):
    r=requests.post('https://api.line.me/v2/bot/message/push',headers={'Authorization':f'Bearer {TOKEN}','Content-Type':'application/json'},json={'to':USER_ID,'messages':[{'type':'text','text':text}]},timeout=15); r.raise_for_status()

def scan_asset(name,symbol):
    print(f"[SCAN] {name} ({symbol})")
    fr={}
    for label in TF:
        fr[label]=ind(fetch(symbol,TF[label]))
    d1,h4,h1,m15,m5=[state(fr[x]) for x in ('1D','4h','1h','15m','5m')]
    print(f"[STATE] {name}: D1={d1} H4={h4} H1={h1} M15={m15} M5={m5}")
    if d1[0] not in ('LONG','SHORT') or not (h4[0]==h1[0]==m15[0]==d1[0]) or d1[1]<60: return []
    direction=d1[0]
    ok_setup,s_score,s_name=setup(fr['15m'],direction)
    ok_trig,t_score=trigger(fr['5m'],direction)

    print(f"[CHECK] {name}: setup={ok_setup} score={s_score} name={s_name} trigger={ok_trig} score={t_score}")

    alerts=[]

    for mode in ('Short Hold','Long Hold'):
        ready=ok_setup and ok_trig

        p=plan(fr,direction,mode) if ready else None

        if p:
            entry,sl,tp1,tp2=p
            candle=str(fr['5m'].index[-1])

            sig=f'{symbol}|{mode}|{direction}|{candle}|{round(entry,4)}|{round(sl,4)}'

            msg=(
                f'🚨 {name} | {mode}\n'
                f'{direction} — ENTRY READY\n'
                f'Entry: {entry:.4f}\n'
                f'SL: {sl:.4f}\n'
                f'TP1: {tp1:.4f}\n'
                f'TP2: {tp2:.4f}\n'
                f'Setup: {s_name} ({s_score})\n'
                f'Trigger: {t_score}'
            )

            alerts.append((sig,msg))

    if alerts:
        print(f"[RESULT] {name}: {len(alerts)} ENTRY READY")
    else:
        print(f"[RESULT] {name}: NO ENTRY")

    return alerts

def main():
    if not all([API_KEY,TOKEN,USER_ID]): raise SystemExit('Missing TWELVEDATA_API_KEY / LINE_CHANNEL_ACCESS_TOKEN / LINE_USER_ID')
    state=load_state(); changed=False
    for name,symbol in ASSETS.items():
        try: alerts=scan_asset(name,symbol)
        except Exception as e: print(f'{name}: {e}'); continue
        for sig,msg in alerts:
            if sig not in state:
                push(msg); state[sig]=True; changed=True; print('sent',sig)
    # Keep state bounded.
    if len(state)>500: state=dict(list(state.items())[-300:]); changed=True
    if changed: save_state(state)

if __name__=='__main__': main()
