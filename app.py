import os, time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title='Gold & Bitcoin Trading Analyzer V4.7', page_icon='📈', layout='wide')
BASE='https://api.twelvedata.com'
ASSETS={'Bitcoin BTC/USD':'BTC/USD','Gold XAU/USD':'XAU/USD'}
TF={'5m':'5min','15m':'15min','1h':'1h','4h':'4h','1D':'1day'}
MIN_CALL_INTERVAL=1.1
CACHE_TTL=60
_last_call_ts=0.0
_LAST_GOOD={}
API_KEY=os.getenv('TWELVEDATA_API_KEY','')
try:
    if not API_KEY and 'TWELVEDATA_API_KEY' in st.secrets: API_KEY=str(st.secrets['TWELVEDATA_API_KEY'])
except Exception: pass

def _pace_requests():
    global _last_call_ts
    wait=MIN_CALL_INTERVAL-(time.monotonic()-_last_call_ts)
    if wait>0: time.sleep(wait)
    _last_call_ts=time.monotonic()

def api_get(endpoint,params):
    if not API_KEY: return None,'ยังไม่ได้ตั้ง TWELVEDATA_API_KEY'
    last='Twelve Data API error'
    for _ in range(2):
        _pace_requests()
        try:
            r=requests.get(f'{BASE}/{endpoint}',params=params,timeout=20); data=r.json()
        except Exception as e:
            last=f'เชื่อมต่อ Twelve Data ไม่สำเร็จ: {e}'; time.sleep(1.5); continue
        if r.status_code==429:
            last='Twelve Data rate limit (free plan) — คำขอถี่เกินไป'; time.sleep(3); continue
        if r.status_code!=200 or ('code' in data and data.get('code',200)>=400): return None,data.get('message','Twelve Data API error')
        return data,None
    return None,last

@st.cache_data(ttl=CACHE_TTL,show_spinner=False)
def _fetch_ohlcv(symbol,interval,outputsize):
    data,err=api_get('time_series',{'symbol':symbol,'interval':interval,'outputsize':outputsize,'apikey':API_KEY,'timezone':'UTC'})
    if err:return pd.DataFrame(),err
    if not data or 'values' not in data:return pd.DataFrame(),(data or {}).get('message','ไม่มีข้อมูล')
    df=pd.DataFrame(data['values'])
    for c in ['open','high','low','close','volume']:
        if c in df: df[c]=pd.to_numeric(df[c],errors='coerce')
    df['datetime']=pd.to_datetime(df['datetime'],utc=True)
    df=df.sort_values('datetime').set_index('datetime')
    return df.dropna(subset=['open','high','low','close']),None

def get_ohlcv(symbol,interval,outputsize=500):
    key=(symbol,interval); df,err=_fetch_ohlcv(symbol,interval,outputsize)
    if err or df.empty:
        cached=_LAST_GOOD.get(key)
        if cached is not None and not cached.empty:return cached,f'{err or "ไม่มีข้อมูลใหม่"} (ใช้ข้อมูลล่าสุดที่แคชไว้)'
        return df,err
    _LAST_GOOD[key]=df; return df,None

def indicators(df):
    d=df.copy()
    if d.empty:return d
    c,h,l=d.close,d.high,d.low
    for n in (20,50,200):d[f'EMA{n}']=c.ewm(span=n,adjust=False).mean()
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    d['ATR']=tr.ewm(alpha=1/14,adjust=False).mean(); d['range']=h-l; d['body']=(c-d.open).abs(); d['body_ratio']=d.body/d['range'].replace(0,np.nan)
    d['upper_wick']=h-d[['open','close']].max(axis=1); d['lower_wick']=d[['open','close']].min(axis=1)-l
    if 'volume' in d:
        d['VolMA20']=d.volume.rolling(20).mean(); d['VolRatio']=d.volume/d.VolMA20.replace(0,np.nan)
    else:d['VolMA20']=d['VolRatio']=np.nan
    return d

def closed_only(d): return d if d is None or len(d)<3 else d.iloc[:-1].copy()

def structure(d,lookback=20):
    if d is None or len(d)<lookback+8:return 'MIXED'
    r,p=d.iloc[-6:],d.iloc[-lookback:-6]; rh,rl=float(r.high.max()),float(r.low.min()); ph,pl=float(p.high.max()),float(p.low.min())
    if rh>ph and rl>pl:return 'HH_HL'
    if rh<ph and rl<pl:return 'LH_LL'
    x=d.iloc[-1]
    if x.EMA20>x.EMA50>x.EMA200:return 'BULL'
    if x.EMA20<x.EMA50<x.EMA200:return 'BEAR'
    return 'MIXED'

def tf_state(d):
    if d is None or len(d)<205:return {'bias':'NEUTRAL','score':50,'structure':'MIXED','ema':'MIXED','slope':0.0}
    x=d.iloc[-1]; stc=structure(d); eb=x.EMA20>x.EMA50>x.EMA200; es=x.EMA20<x.EMA50<x.EMA200; slope=float(x.EMA20-d.EMA20.iloc[-6]); cb=x.close>x.EMA20; cs=x.close<x.EMA20
    bull=int(eb)+int(cb)+int(slope>0)+2*int(stc in ('HH_HL','BULL')); bear=int(es)+int(cs)+int(slope<0)+2*int(stc in ('LH_LL','BEAR')); diff=bull-bear
    return {'bias':'LONG' if diff>0 else 'SHORT' if diff<0 else 'NEUTRAL','score':int(np.clip(50+abs(diff)*10,0,100)),'structure':stc,'ema':'LONG' if eb else 'SHORT' if es else 'MIXED','slope':slope}

def range_info(d,n=48):
    x=d.tail(min(n,len(d))); hi,lo=float(x.high.max()),float(x.low.min()); span=max(hi-lo,1e-9); pos=(float(d.close.iloc[-1])-lo)/span
    return {'high':hi,'low':lo,'span':span,'pos':pos,'zone':'LOWER RANGE' if pos<.33 else 'UPPER RANGE' if pos>.67 else 'MID RANGE'}

def macro_context(d1,h4,h1,m15):
    s1,s4,sh,s15=[tf_state(x) for x in (d1,h4,h1,m15)]
    if s1['bias']==s4['bias'] and s1['bias'] in ('LONG','SHORT'): macro=s1['bias']; ms=int(round((s1['score']+s4['score'])/2))
    elif s1['bias'] in ('LONG','SHORT') and s4['bias']=='NEUTRAL': macro=s1['bias']; ms=int(s1['score']*.9)
    elif s4['bias'] in ('LONG','SHORT') and s1['bias']=='NEUTRAL': macro=s4['bias']; ms=int(s4['score']*.9)
    else: macro='NEUTRAL'; ms=int((s1['score']+s4['score'])/2)
    if sh['bias']==s15['bias'] and sh['bias'] in ('LONG','SHORT'): tactical=sh['bias']; ts=int(round((sh['score']+s15['score'])/2))
    elif s15['bias'] in ('LONG','SHORT'): tactical=s15['bias']; ts=s15['score']
    elif sh['bias'] in ('LONG','SHORT'): tactical=sh['bias']; ts=sh['score']
    else: tactical='NEUTRAL'; ts=int((sh['score']+s15['score'])/2)
    return {'d1':s1,'h4':s4,'h1':sh,'m15':s15,'macro':macro,'macro_strength':ms,'tactical':tactical,'tactical_strength':ts}

def _m15_candidate(row,tactical,d):
    atr=max(float(row.ATR),1e-9); rg=max(float(row.range),1e-9); e20,e50=float(row.EMA20),float(row.EMA50); hi20=float(d.iloc[-21:-1].high.max()); lo20=float(d.iloc[-21:-1].low.min()); rl=float(d.iloc[-8:-1].low.min()); rh=float(d.iloc[-8:-1].high.max())
    bp=row.low<=e20+.75*atr and row.close>=e20 and row.close>row.open and row.close>=row.low+.50*rg; sp=row.low<rl and row.close>rl and row.close>row.open; bb=row.close>hi20 and row.close>row.open and row.body_ratio>=.30; bc=row.close>e20 and e20>=e50*.998 and row.low>float(d.iloc[-4:-1].low.min())
    br=row.high>=e20-.75*atr and row.close<=e20 and row.close<row.open and row.close<=row.high-.50*rg; sr=row.high>rh and row.close<rh and row.close<row.open; bd=row.close<lo20 and row.close<row.open and row.body_ratio>=.30; sc=row.close<e20 and e20<=e50*1.002 and row.high<float(d.iloc[-4:-1].high.max())
    if tactical=='LONG':return [n for ok,n in ((bp,'M15 pullback'),(sp,'M15 sweep-reclaim'),(bb,'M15 breakout'),(bc,'M15 continuation')) if ok]
    return [n for ok,n in ((br,'M15 pullback'),(sr,'M15 sweep-reject'),(bd,'M15 breakdown'),(sc,'M15 continuation')) if ok]

def m15_setup(d,direction):
    if d is None or len(d)<80 or direction=='NEUTRAL':return {'ok':False,'near':False,'score':0,'name':'รอ M15 setup','type':'NONE'}
    c=[]
    for off in (0,1,2):
        sub=d if off==0 else d.iloc[:-off]
        if len(sub)<30:continue
        for n in _m15_candidate(sub.iloc[-1],direction,sub):c.append((off,n))
    if c:
        newest=min(o for o,_ in c); names=[]
        for off,n in c:
            if off==newest and n not in names:names.append(n)
        return {'ok':True,'near':False,'score':min(100,62+8*len(names)),'name':' + '.join(names),'type':names[0]}
    x=d.iloc[-1]; atr=max(float(x.ATR),1e-9); near=abs(float(x.close)-float(x.EMA20))/atr<=1.25
    return {'ok':False,'near':near,'score':52 if near else 35,'name':'M15 ใกล้ setup zone' if near else 'รอ M15 pullback / breakout / sweep / continuation','type':'NEAR' if near else 'NONE'}

def m5_trigger(d,direction):
    if d is None or len(d)<50 or direction=='NEUTRAL':return {'ok':False,'score':0,'name':'รอ M5 trigger','type':'NONE'}
    candidates=[]
    for off in (0,1,2):
        sub=d if off==0 else d.iloc[:-off]
        if len(sub)<25:continue
        x,p=sub.iloc[-1],sub.iloc[-2]; rg=max(float(x.range),1e-9)
        bb=x.close>p.high and x.close>x.open and x.body_ratio>=.20; bd=x.close<p.low and x.close<x.open and x.body_ratio>=.20
        br=x.close>x.EMA20 and p.close<=p.EMA20 and x.close>x.open; sr=x.close<x.EMA20 and p.close>=p.EMA20 and x.close<x.open
        bj=x.lower_wick>=.20*rg and x.close>x.open and x.close>=x.low+.55*rg; sj=x.upper_wick>=.20*rg and x.close<x.open and x.close<=x.high-.55*rg
        if direction=='LONG':score=min(100,40*int(bb)+35*int(br)+25*int(bj)); name='M5 bullish trigger'
        else:score=min(100,40*int(bd)+35*int(sr)+25*int(sj)); name='M5 bearish trigger'
        candidates.append((off,score,name))
    q=[c for c in candidates if c[1]>=55]
    if q:
        _,score,name=min(q,key=lambda c:c[0]); return {'ok':True,'score':int(score),'name':name,'type':direction}
    _,score,_=max(candidates,key=lambda c:c[1]); return {'ok':False,'score':int(score),'name':'รอ M5 กลับขึ้น' if direction=='LONG' else 'รอ M5 กลับลง','type':'NONE'}

def scan_signals(d,threshold=55,min_gap=4):
    buys=[]; sells=[]; last_state=None; last_i=-10**9
    if d is None or len(d)<2:return buys,sells
    for i in range(1,len(d)):
        x,p=d.iloc[i],d.iloc[i-1]; rg=max(float(x.range),1e-9)
        bb=x.close>p.high and x.close>x.open and x.body_ratio>=.20; bd=x.close<p.low and x.close<x.open and x.body_ratio>=.20; br=x.close>x.EMA20 and p.close<=p.EMA20 and x.close>x.open; sr=x.close<x.EMA20 and p.close>=p.EMA20 and x.close<x.open; bj=x.lower_wick>=.20*rg and x.close>x.open and x.close>=x.low+.55*rg; sj=x.upper_wick>=.20*rg and x.close<x.open and x.close<=x.high-.55*rg
        ls=min(100,40*int(bb)+35*int(br)+25*int(bj)+10*int(np.isfinite(x.VolRatio) and x.VolRatio>=1)+20*int(x.close>x.EMA20)); ss=min(100,40*int(bd)+35*int(sr)+25*int(sj)+10*int(np.isfinite(x.VolRatio) and x.VolRatio>=1)+20*int(x.close<x.EMA20))
        state='LONG' if ls>=threshold and ls>=ss else 'SHORT' if ss>=threshold else None
        if i-last_i>=min_gap:
            if state=='LONG' and last_state!='LONG':buys.append({'time':x.name,'price':float(x.low),'score':int(ls)});last_i=i
            elif state=='SHORT' and last_state!='SHORT':sells.append({'time':x.name,'price':float(x.high),'score':int(ss)});last_i=i
        last_state=state
    return buys,sells

def range_location(h4,h1,direction):
    if direction not in ('LONG','SHORT'):return {'score':50,'zone':'MIXED','reason':'ยังไม่มี direction'}
    z=[range_info(h4)['zone'],range_info(h1)['zone']]
    if direction=='LONG':
        if z.count('LOWER RANGE')==2:return {'score':90,'zone':'LOWER RANGE','reason':'H4/H1 อยู่โซนล่าง เหมาะกับการหาจังหวะ Long'}
        if 'LOWER RANGE' in z:return {'score':75,'zone':'LOWER/MID','reason':'มีอย่างน้อยหนึ่ง TF อยู่โซนล่าง'}
        if z.count('UPPER RANGE')==2:return {'score':30,'zone':'UPPER RANGE','reason':'ราคาอยู่โซนบนของ H4/H1 ไม่เหมาะกับการไล่ Long'}
        return {'score':55,'zone':'MID RANGE','reason':'H4/H1 อยู่กลาง range'}
    if z.count('UPPER RANGE')==2:return {'score':90,'zone':'UPPER RANGE','reason':'H4/H1 อยู่โซนบน เหมาะกับการหาจังหวะ Short'}
    if 'UPPER RANGE' in z:return {'score':75,'zone':'UPPER/MID','reason':'มีอย่างน้อยหนึ่ง TF อยู่โซนบน'}
    if z.count('LOWER RANGE')==2:return {'score':30,'zone':'LOWER RANGE','reason':'ราคาอยู่โซนล่าง ไม่เหมาะกับการไล่ Short'}
    return {'score':55,'zone':'MID RANGE','reason':'H4/H1 อยู่กลาง range'}

def make_plan(frames,direction,mode):
    m5,m15,h1=frames['5m'],frames['15m'],frames['1h']
    if direction=='NEUTRAL':return None
    atr5=max(float(m5.ATR.iloc[-1]),1e-9); atr15=max(float(m15.ATR.iloc[-1]),1e-9); entry=float(m5.close.iloc[-1])
    if mode=='Short Hold':
        risk_min,risk_max=atr5,2.8*atr5
        if direction=='LONG':sl=float(m5.tail(12).low.min())-.25*atr5; risk=entry-sl
        else:sl=float(m5.tail(12).high.max())+.25*atr5; risk=sl-entry
        if 0<risk<risk_min:risk=risk_min;sl=entry-risk if direction=='LONG' else entry+risk
        if risk<=0 or risk>risk_max:return None
        tp1,tp2=(entry+1.2*risk,entry+1.8*risk) if direction=='LONG' else (entry-1.2*risk,entry-1.8*risk)
    else:
        entry=float(m15.close.iloc[-1]); risk_min,risk_max=1.2*atr15,5*atr15
        if direction=='LONG':sl=min(float(m15.tail(14).low.min()),float(h1.tail(10).low.min()))-.30*atr15; risk=entry-sl
        else:sl=max(float(m15.tail(14).high.max()),float(h1.tail(10).high.max()))+.30*atr15; risk=sl-entry
        if 0<risk<risk_min:risk=risk_min;sl=entry-risk if direction=='LONG' else entry+risk
        if risk<=0 or risk>risk_max:return None
        tp1,tp2=(entry+1.5*risk,entry+3*risk) if direction=='LONG' else (entry-1.5*risk,entry-3*risk)
    return {'entry':entry,'sl':sl,'tp1':tp1,'tp2':tp2,'risk':risk,'rr':abs(tp2-entry)/risk}

def evaluate(frames,mode):
    ctx=macro_context(frames['1D'],frames['4h'],frames['1h'],frames['15m']); macro=ctx['macro']; tactical=ctx['tactical']; MIN_MACRO_SCORE=60
    # Trend-following gate: D1/H4/H1/M15 must all agree before ENTRY READY.
    full_align=(macro in ('LONG','SHORT') and ctx['d1']['bias']==macro and ctx['h4']['bias']==macro and ctx['h1']['bias']==macro and ctx['m15']['bias']==macro)
    counter=macro in ('LONG','SHORT') and tactical in ('LONG','SHORT') and tactical!=macro
    short_dir=macro; short_loc=range_location(frames['4h'],frames['1h'],short_dir); short_setup=m15_setup(frames['15m'],short_dir); short_trig=m5_trigger(frames['5m'],short_dir)
    long_dir=macro; long_loc=range_location(frames['4h'],frames['1h'],long_dir); long_setup=m15_setup(frames['15m'],long_dir); long_trig=m5_trigger(frames['5m'],long_dir)
    if mode=='Short Hold':
        direction,setup,trig,loc=short_dir,short_setup,short_trig,short_loc
        if macro=='NEUTRAL':status='WAIT';entry_class='WAIT';reason='D1/H4 ยังไม่ให้ Macro direction'
        elif ctx['macro_strength']<MIN_MACRO_SCORE:status='WAIT';entry_class='WAIT';reason=f'Macro Strength {ctx["macro_strength"]}/100 ต่ำกว่าเกณฑ์ {MIN_MACRO_SCORE} — รอ Macro แข็งแรงขึ้น'
        elif not full_align:status='PRE-ENTRY';entry_class='TREND WATCH';reason=f'Macro {macro} แต่ D1/H4/H1/M15 ยังไม่ align ครบ — รอทุก TF กลับ {macro}'
        elif setup['ok'] and trig['ok']:status='ENTRY READY';entry_class='TREND ENTRY';reason=f'Trend aligned • {setup["name"]} • {trig["name"]} • {loc["zone"]}'
        elif setup['ok'] or setup['near']:status='PRE-ENTRY';entry_class='TREND WATCH';reason=f'{setup["name"]} • {trig["name"]} • {loc["reason"]}'
        else:status='WAIT';entry_class='WAIT';reason=setup['name']
    else:
        direction,setup,trig,loc=long_dir,long_setup,long_trig,long_loc
        if macro=='NEUTRAL':status='WAIT';entry_class='WAIT';reason='D1/H4 ยังไม่ให้ Macro direction'
        elif ctx['macro_strength']<MIN_MACRO_SCORE:status='WAIT';entry_class='WAIT';reason=f'Macro Strength {ctx["macro_strength"]}/100 ต่ำกว่าเกณฑ์ {MIN_MACRO_SCORE} — รอ Macro แข็งแรงขึ้น'
        elif not full_align:status='PRE-ENTRY' if (setup['ok'] or setup['near']) else 'WAIT';entry_class='PULLBACK / RESUME';reason=f'Macro {macro} • Tactical {tactical} • รอ D1/H4/H1/M15 align และ M5 trigger • {setup["name"]} • {trig["name"]}'
        elif setup['ok'] and trig['ok']:status='ENTRY READY';entry_class='TREND ENTRY';reason=f'Trend aligned • {setup["name"]} • {trig["name"]} • {loc["zone"]}'
        elif setup['ok'] or setup['near']:status='PRE-ENTRY';entry_class='TREND WATCH';reason=f'{setup["name"]} • {trig["name"]} • {loc["reason"]}'
        else:status='WAIT';entry_class='WAIT';reason=setup['name']
    readiness=int(np.clip(.30*ctx['macro_strength']+.25*ctx['tactical_strength']+.20*setup['score']+.15*trig['score']+.10*loc['score'],0,100))
    plan=make_plan(frames,direction,mode) if status=='ENTRY READY' else None
    if status=='ENTRY READY' and plan is None:status='PRE-ENTRY';entry_class='TREND WATCH';reason='สัญญาณครบ แต่โครงสร้าง SL กว้างเกินไป — รอราคาเข้าโครงสร้างใหม่'
    relation='ALIGNED' if full_align else 'COUNTER-MACRO' if counter else 'MIXED'
    return {'status':status,'direction':direction,'macro':macro,'tactical':tactical,'macro_strength':ctx['macro_strength'],'tactical_strength':ctx['tactical_strength'],'relation':relation,'entry_class':entry_class,'trend_d1':ctx['d1']['score'],'h4_score':ctx['h4']['score'],'h1_score':ctx['h1']['score'],'alignment':int(round((ctx['h4']['score']+ctx['h1']['score'])/2)),'setup':setup,'trigger':trig,'range':range_info(frames['15m']),'location':loc,'readiness':readiness,'plan':plan,'reason':reason,'states':ctx}

def fmt(v):return '—' if v is None or not np.isfinite(v) else f'{v:,.2f}'

def render_card(r,title):
    ready=r['status']=='ENTRY READY'; pre=r['status']=='PRE-ENTRY'; icon='🟢' if ready and r['direction']=='LONG' else '🔴' if ready and r['direction']=='SHORT' else '🟡'
    st.markdown(f'### {title}'); st.markdown(f'**{icon} {r["status"]} — {r["direction"]}**')
    if r['entry_class']=='PULLBACK / RESUME':st.info('PULLBACK / RESUME — Macro กับ Tactical/TF ย่อยยังไม่ align ครบ จึงยังไม่ถือเป็น entry')
    elif r['entry_class']=='TREND ENTRY':st.caption('TREND ENTRY — D1/H4/H1/M15 aligned และ M15/M5 confirmation ครบ')
    a,b,c=st.columns(3); a.metric('Macro D1/H4',f'{r["macro"]} {r["macro_strength"]}/100'); b.metric('Tactical H1/M15',f'{r["tactical"]} {r["tactical_strength"]}/100'); c.metric('Setup Readiness',f'{r["readiness"]}/100')
    st.caption(f'Signal Class: {r["entry_class"]} • Macro {r["macro"]} • Tactical {r["tactical"]} • {r["relation"]} • H4/H1 {r["location"]["zone"]} • M15 {r["setup"]["name"]} • M5 {r["trigger"]["name"]}')
    st.info(('PRE-ENTRY: ' if pre else 'ENTRY READY: ' if ready else 'WAIT: ')+r['reason'])
    if ready and r['plan']:
        p=r['plan']; a,b,c,d=st.columns(4); a.metric('Entry',fmt(p['entry'])); b.metric('SL',fmt(p['sl'])); c.metric('TP1',fmt(p['tp1'])); d.metric('TP2',fmt(p['tp2'])); st.caption(f'R:R to TP2 ≈ {p["rr"]:.2f}R')
    else:st.caption('ยังไม่แสดง Entry / SL / TP จนกว่าจะเกิด ENTRY READY')

st.title('Gold & Bitcoin Trading Analyzer — V4.7')
st.caption('D1 trend → H4 → H1 → M15 setup → M5 trigger. Short Hold และ Long Hold แสดงพร้อมกัน • Trend-following only')
with st.sidebar:
    st.header('ตั้งค่าการวิเคราะห์'); asset_name=st.selectbox('สินทรัพย์',list(ASSETS.keys()),index=0); symbol=ASSETS[asset_name]; chart_tf=st.selectbox('Timeframe กราฟ',list(TF.keys()),index=1); outputsize=st.slider('จำนวนแท่ง',250,800,400,50); auto=st.checkbox('Auto refresh',False); refresh=st.slider('รอบรีเฟรช (วินาที)',60,300,90,10)
    if st.button('รีเฟรชข้อมูลตอนนี้'):st.cache_data.clear();st.rerun()
    st.divider(); show_signals=st.checkbox('แสดงจุด BUY/SELL บนกราฟ (ย้อนหลัง)',True); signal_threshold=st.slider('เกณฑ์คะแนนสัญญาณ',40,90,55,5) if show_signals else 55; signal_gap=st.slider('ระยะห่างขั้นต่ำระหว่างจุดสัญญาณ (แท่ง)',1,10,4,1) if show_signals else 4; show_zone=st.checkbox('แสดงโซน pullback',True); st.caption(f'API: Twelve Data • cache {CACHE_TTL}s • 5 requests/load')
if auto:st.markdown(f'<meta http-equiv="refresh" content="{refresh}">',unsafe_allow_html=True)
frames,raws,errors,stale={},{},{},{}
for label in ['1D','4h','1h','15m','5m']:
    raw,err=get_ohlcv(symbol,TF[label],outputsize); raws[label]=raw if raw is not None else pd.DataFrame(); frames[label]=indicators(closed_only(raws[label])) if not raws[label].empty else pd.DataFrame()
    if err:errors[label]=err; stale[label]=err if raw is not None and not raw.empty else None
chart=frames.get(chart_tf,pd.DataFrame()); have_all=all(not frames[x].empty for x in ['1D','4h','1h','15m','5m'])
if errors:
    hard={k:v for k,v in errors.items() if stale.get(k) is None}; soft={k:v for k,v in errors.items() if stale.get(k) is not None}
    if soft:st.warning('บางไทม์เฟรมใช้ข้อมูลล่าสุดที่แคชไว้: '+' | '.join(f'{k}: {v}' for k,v in soft.items()))
    if hard:st.error(' | '.join(f'{k}: {v}' for k,v in hard.items()))
if chart.empty:st.error('ไม่มีข้อมูลสำหรับกราฟ/ราคาปัจจุบัน');st.stop()
price_src=frames.get('5m') if not frames.get('5m',pd.DataFrame()).empty else chart; price=float(price_src.close.iloc[-1]); prev=float(price_src.close.iloc[-2]) if len(price_src)>1 else price; st.subheader(asset_name);st.metric('Price',fmt(price),f'{(price/prev-1)*100:+.2f}%' if prev else '0.00%')
short=long=None
st.markdown('## สถานะการเทรด')
if have_all:
    short=evaluate(frames,'Short Hold');long=evaluate(frames,'Long Hold');a,b=st.columns(2)
    with a:render_card(short,'Short Hold')
    with b:render_card(long,'Long Hold')
else:st.info('ข้อมูลบางไทม์เฟรมยังไม่พร้อม — รอแล้วรีเฟรชใหม่')
st.markdown('## โครงสร้างตลาด'); rows=[]
for tf in ['1D','4h','1h','15m','5m']:
    d=frames[tf]
    if d.empty:rows.append({'TF':tf,'Bias':'—','Trend':'—','Structure':'—','Range':'—','Range High':'—','Range Low':'—'});continue
    s=tf_state(d);rg=range_info(d);rows.append({'TF':tf,'Bias':s['bias'],'Trend':s['score'],'Structure':s['structure'],'Range':rg['zone'],'Range High':rg['high'],'Range Low':rg['low']})
st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
st.markdown(f'## กราฟ {chart_tf}');fig=go.Figure();fig.add_trace(go.Candlestick(x=chart.index,open=chart.open,high=chart.high,low=chart.low,close=chart.close,name='Price'))
for n in (20,50,200):fig.add_trace(go.Scatter(x=chart.index,y=chart[f'EMA{n}'],name=f'EMA{n}',mode='lines'))
if show_signals:
    buys,sells=scan_signals(chart,signal_threshold,signal_gap)
    if buys:fig.add_trace(go.Scatter(x=[x['time'] for x in buys],y=[x['price'] for x in buys],mode='markers',name='BUY',marker=dict(symbol='triangle-up',size=11,color='#22c55e'),text=[f"BUY {x['score']}%" for x in buys],hovertemplate='%{text}<br>%{y}<extra></extra>'))
    if sells:fig.add_trace(go.Scatter(x=[x['time'] for x in sells],y=[x['price'] for x in sells],mode='markers',name='SELL',marker=dict(symbol='triangle-down',size=11,color='#ef4444'),text=[f"SELL {x['score']}%" for x in sells],hovertemplate='%{text}<br>%{y}<extra></extra>'))
if show_zone and len(chart)>20 and short and long:
    zones=[]
    for res,label in ((short,'Short Hold'),(long,'Long Hold')):
        if res['status']=='PRE-ENTRY' and res['direction'] in ('LONG','SHORT'):zones.append((res['direction'],label))
    if not zones:
        for res,label in ((short,'Short Hold'),(long,'Long Hold')):
            if res['direction'] in ('LONG','SHORT'):zones.append((res['direction'],label));break
    last=chart.iloc[-1]; e=float(last.EMA20); atr=float(last.ATR); delta=chart.index.to_series().diff().dropna(); step=delta.median() if len(delta) else pd.Timedelta(minutes=15)
    if zones and np.isfinite(e) and np.isfinite(atr) and atr>0:
        for i,(zd,zl) in enumerate(zones):
            x0=chart.index[-1]+step*12*i;x1=chart.index[-1]+step*12*(i+1); lc='#22c55e' if zd=='LONG' else '#ef4444'; fc='rgba(34,197,94,0.16)' if zd=='LONG' else 'rgba(239,68,68,0.16)'; fig.add_shape(type='rect',xref='x',yref='y',x0=x0,x1=x1,y0=e-.75*atr,y1=e+.75*atr,fillcolor=fc,line=dict(width=1,color=lc,dash='dot'),layer='below');fig.add_annotation(x=x1,y=e+.75*atr,text=f'{zl} ({zd})',showarrow=False,xanchor='right',yanchor='bottom',font=dict(size=10,color=lc))
fig.update_layout(height=520,xaxis_rangeslider_visible=False,margin=dict(l=10,r=10,t=30,b=10));st.plotly_chart(fig,use_container_width=True)
if show_signals:st.caption('BUY/SELL บนกราฟเป็น raw candle markers เท่านั้น ไม่ได้ผ่าน Macro/Tactical gate และไม่ส่งคำสั่งจริง')
st.markdown('## หลักการของ V4.7');st.write('Trend-following only: D1/H4/H1/M15 ต้อง align กันก่อน ENTRY READY และ Macro Strength ต้อง ≥60. M15 เป็น setup และ M5 เป็น trigger. M5 trigger ใช้เฉพาะ breakout/reclaim/rejection points; ไม่มีคะแนนฟรีจาก volume หรือ EMA. Short Hold ไม่มี COUNTER-MACRO ENTRY. Long Hold ที่ TF ย่อยยังไม่ align จะแสดง PULLBACK / RESUME และยังไม่เป็น ENTRY READY. ระบบเป็น analyzer ไม่ส่งคำสั่งซื้อขายอัตโนมัติ')
