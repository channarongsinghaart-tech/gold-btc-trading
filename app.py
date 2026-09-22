import os
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title='Gold & Bitcoin Trading Analyzer V4.3', page_icon='📈', layout='wide')

BASE = 'https://api.twelvedata.com'
ASSETS = {'Bitcoin BTC/USD':'BTC/USD','Gold XAU/USD':'XAU/USD'}
TF = {'5m':'5min','15m':'15min','30m':'30min','1h':'1h','4h':'4h','1D':'1day'}

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
        r=requests.get(f'{BASE}/{endpoint}',params=params,timeout=20)
        data=r.json()
    except Exception as exc:
        return None, f'เชื่อมต่อ Twelve Data ไม่สำเร็จ: {exc}'
    if r.status_code != 200 or ('code' in data and data.get('code',200) >= 400):
        return None, data.get('message','Twelve Data API error')
    return data,None


@st.cache_data(ttl=45, show_spinner=False)
def get_ohlcv(symbol, interval, outputsize=500):
    data,err=api_get('time_series',{
        'symbol':symbol,'interval':interval,'outputsize':outputsize,
        'apikey':API_KEY,'timezone':'UTC'
    })
    if err: return pd.DataFrame(),err
    if not data or 'values' not in data:
        return pd.DataFrame(),(data or {}).get('message','ไม่มีข้อมูล')
    df=pd.DataFrame(data['values'])
    for c in ['open','high','low','close','volume']:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors='coerce')
    df['datetime']=pd.to_datetime(df['datetime'],utc=True)
    df=df.sort_values('datetime').set_index('datetime')
    return df.dropna(subset=['open','high','low','close']),None


def indicators(df):
    d=df.copy(); c,h,l=d.close,d.high,d.low
    for n in (20,50,200): d[f'EMA{n}']=c.ewm(span=n,adjust=False).mean()
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    d['ATR']=tr.ewm(alpha=1/14,adjust=False).mean()
    delta=c.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    rs=gain/loss.replace(0,np.nan); d['RSI']=100-(100/(1+rs))
    d['range']=h-l; d['body']=(c-d.open).abs(); d['body_ratio']=d.body/d['range'].replace(0,np.nan)
    d['upper_wick']=h-d[['open','close']].max(axis=1)
    d['lower_wick']=d[['open','close']].min(axis=1)-l
    if 'volume' in d.columns:
        d['VolMA20']=d.volume.rolling(20).mean(); d['VolRatio']=d.volume/d.VolMA20
    else: d['VolMA20']=np.nan; d['VolRatio']=np.nan
    return d


def structure(d, n=20):
    if len(d)<n+6: return 'MIXED'
    recent=d.iloc[-6:]; prior=d.iloc[-n:-6]
    rh,rl=float(recent.high.max()),float(recent.low.min())
    ph,pl=float(prior.high.max()),float(prior.low.min())
    if rh>ph and rl>pl: return 'HH_HL'
    if rh<ph and rl<pl: return 'LH_LL'
    x=d.iloc[-1]
    if x.EMA20>x.EMA50>x.EMA200: return 'BULL'
    if x.EMA20<x.EMA50<x.EMA200: return 'BEAR'
    return 'MIXED'


def trend_score(d):
    if d is None or len(d)<205: return 'NEUTRAL',50,{'structure':'MIXED','ema':0,'price':0,'slope':0}
    x=d.iloc[-1]; atr=max(float(x.ATR),1e-9); st=structure(d)
    bull=bear=0
    bull += 1 if x.EMA20>x.EMA50 else 0; bear += 1 if x.EMA20<x.EMA50 else 0
    bull += 1 if x.EMA50>x.EMA200 else 0; bear += 1 if x.EMA50<x.EMA200 else 0
    bull += 1 if x.close>x.EMA20 else 0; bear += 1 if x.close<x.EMA20 else 0
    slope=x.EMA20-d.EMA20.iloc[-6]
    bull += 1 if slope>0 else 0; bear += 1 if slope<0 else 0
    bull += 2 if st in ('HH_HL','BULL') else 0; bear += 2 if st in ('LH_LL','BEAR') else 0
    diff=bull-bear
    if diff>0: bias='LONG'
    elif diff<0: bias='SHORT'
    else: bias='NEUTRAL'
    score=int(np.clip(50+abs(diff)*10,0,100))
    return bias,score,{'structure':st,'ema':bull-bear,'price':1 if x.close>x.EMA20 else -1,'slope':float(slope/atr)}


def range_info(d,n=48):
    x=d.tail(n)
    hi=float(x.high.max()); lo=float(x.low.min()); span=max(hi-lo,1e-9); price=float(d.close.iloc[-1])
    pos=(price-lo)/span
    if pos<0.20: zone='LOWER RANGE'
    elif pos>0.80: zone='UPPER RANGE'
    else: zone='MID RANGE'
    return {'high':hi,'low':lo,'span':span,'pos':pos,'zone':zone}


def entry_patterns(d, direction):
    if d is None or len(d)<60: return {'setup':False,'name':'ข้อมูลไม่พอ','score':0,'reasons':[]}
    x=d.iloc[-1]; p=d.iloc[-2]; atr=max(float(x.ATR),1e-9); rg=max(float(x.range),1e-9)
    recent_hi=float(d.iloc[-21:-1].high.max()); recent_lo=float(d.iloc[-21:-1].low.min())
    bull_break=x.close>recent_hi and x.close>x.open and x.body_ratio>=0.45 and x.body>=0.25*atr
    bear_break=x.close<recent_lo and x.close<x.open and x.body_ratio>=0.45 and x.body>=0.25*atr
    bull_pull=x.close>x.EMA20 and x.EMA20>=x.EMA50 and x.low<=x.EMA20+0.25*atr and x.close>x.open
    bear_pull=x.close<x.EMA20 and x.EMA20<=x.EMA50 and x.high>=x.EMA20-0.25*atr and x.close<x.open
    bull_sweep=x.low<float(d.iloc[-8:-1].low.min()) and x.close>float(d.iloc[-8:-1].low.min()) and x.close>x.open
    bear_sweep=x.high>float(d.iloc[-8:-1].high.max()) and x.close<float(d.iloc[-8:-1].high.max()) and x.close<x.open
    names=[]
    if direction=='LONG':
        if bull_pull: names.append('M15 pullback')
        if bull_break: names.append('M15 breakout')
        if bull_sweep: names.append('M15 sweep-reclaim')
    else:
        if bear_pull: names.append('M15 pullback')
        if bear_break: names.append('M15 breakdown')
        if bear_sweep: names.append('M15 sweep-reject')
    return {'setup':bool(names),'name':' + '.join(names) if names else 'รอ M15 setup','score':min(100,35+25*len(names)),'reasons':names}


def m5_trigger(d,direction):
    if d is None or len(d)<30: return {'ok':False,'score':50,'name':'M5 data unavailable'}
    x=d.iloc[-1]; p=d.iloc[-2]; atr=max(float(x.ATR),1e-9); rg=max(float(x.range),1e-9)
    bull=(x.close>x.open and x.close>p.high and x.body_ratio>=0.35) or (x.close>x.open and x.lower_wick>=0.35*rg and x.close>=x.low+0.65*rg)
    bear=(x.close<x.open and x.close<p.low and x.body_ratio>=0.35) or (x.close<x.open and x.upper_wick>=0.35*rg and x.close<=x.high-0.65*rg)
    vr=float(x.VolRatio) if np.isfinite(x.VolRatio) else 1.0
    score=50
    if direction=='LONG':
        if bull: score+=30
        if x.close>x.EMA20: score+=10
        if vr>=1.1: score+=10
        ok=score>=70
        name='M5 bullish trigger' if ok else 'รอ M5 ยืนยัน Long'
    else:
        if bear: score+=30
        if x.close<x.EMA20: score+=10
        if vr>=1.1: score+=10
        ok=score>=70
        name='M5 bearish trigger' if ok else 'รอ M5 ยืนยัน Short'
    return {'ok':ok,'score':int(np.clip(score,0,100)),'name':name}


def make_plan(m15,direction,h1,h4,mode):
    x=m15.iloc[-1]; atr=max(float(x.ATR),1e-9); price=float(x.close)
    r=range_info(m15,48)
    if direction=='LONG':
        structure_low=min(float(m15.tail(12).low.min()),float(h1.tail(12).low.min()),float(h4.tail(12).low.min()))
        sl=min(structure_low,price-1.2*atr)
        if sl>=price: sl=price-1.2*atr
        risk=price-sl; tp1=price+(1.25 if mode=='Short Hold' else 1.5)*risk; tp2=price+(2.0 if mode=='Short Hold' else 3.0)*risk
    else:
        structure_high=max(float(m15.tail(12).high.max()),float(h1.tail(12).high.max()),float(h4.tail(12).high.max()))
        sl=max(structure_high,price+1.2*atr)
        if sl<=price: sl=price+1.2*atr
        risk=sl-price; tp1=price-(1.25 if mode=='Short Hold' else 1.5)*risk; tp2=price-(2.0 if mode=='Short Hold' else 3.0)*risk
    return {'entry':price,'sl':float(sl),'tp1':float(tp1),'tp2':float(tp2),'risk':float(risk),'range_zone':r['zone']}


def evaluate(frames,mode):
    d1,h4,h1,m15,m5=[frames[k] for k in ['1D','4h','1h','15m','5m']]
    b1,s1,i1=trend_score(d1); b4,s4,i4=trend_score(h4); bH,sH,iH=trend_score(h1)
    direction='LONG' if (b1=='LONG' and (b4=='LONG' or bH=='LONG')) else 'SHORT' if (b1=='SHORT' and (b4=='SHORT' or bH=='SHORT')) else 'NEUTRAL'
    if direction=='NEUTRAL':
        if b4=='LONG' and bH=='LONG': direction='LONG'
        elif b4=='SHORT' and bH=='SHORT': direction='SHORT'
    align=int(np.mean([s1,s4,sH]))
    m15range=range_info(m15,48); setup=entry_patterns(m15,direction if direction!='NEUTRAL' else 'LONG') if direction!='NEUTRAL' else {'setup':False,'name':'D1/H4/H1 ยังไม่ชัด','score':25,'reasons':[]}
    trigger=m5_trigger(m5,direction) if direction!='NEUTRAL' else {'ok':False,'score':50,'name':'รอ Direction'}
    # Location score: prefer lower half for longs and upper half for shorts; avoid middle.
    loc_pos=m15range['pos']
    if direction=='LONG': loc=int(np.clip(100-abs(loc_pos-0.30)*120,0,100))
    elif direction=='SHORT': loc=int(np.clip(100-abs(loc_pos-0.70)*120,0,100))
    else: loc=50
    readiness=int(np.clip(0.35*align+0.35*setup['score']+0.20*trigger['score']+0.10*loc,0,100))
    plan=make_plan(m15,direction,h1,h4,mode) if direction!='NEUTRAL' else None
    rr=0 if not plan else (abs(plan['tp2']-plan['entry'])/max(abs(plan['entry']-plan['sl']),1e-9))
    if direction=='NEUTRAL': status='WAIT'
    elif mode=='Short Hold': status='ENTRY READY' if setup['setup'] and trigger['ok'] and align>=55 else 'WAIT'
    else: status='ENTRY READY' if setup['setup'] and align>=55 else 'WAIT'
    if status=='ENTRY READY':
        reason=setup['name'] + (' + '+trigger['name'] if mode=='Short Hold' else '')
    elif direction=='NEUTRAL': reason='รอ D1/H4/H1 ให้ทิศทางชัด'
    elif not setup['setup']: reason='รอ M15 pullback / breakout / sweep'
    elif mode=='Short Hold' and not trigger['ok']: reason='รอ M5 trigger'
    else: reason='รอ alignment เพิ่ม'
    return {'status':status,'direction':direction,'trend_d1':s1,'trend_h4':s4,'trend_h1':sH,'alignment':align,'setup':setup,'trigger':trigger,'range':m15range,'location':loc,'readiness':readiness,'plan':plan,'rr':rr,'reason':reason,'state_d1':i1['structure'],'state_h4':i4['structure'],'state_h1':iH['structure']}


def fmt(v):
    return '—' if v is None or not np.isfinite(v) else f'{v:,.2f}'


def card(result,title):
    status=result['status']; direction=result['direction']
    stt='🟢 ENTRY READY' if status=='ENTRY READY' and direction=='LONG' else '🔴 ENTRY READY' if status=='ENTRY READY' else '🟡 WAIT'
    st.markdown(f'### {title}')
    st.markdown(f'**{stt}** — {direction}')
    a,b,c=st.columns(3)
    a.metric('Trend D1',f"{result['trend_d1']}/100")
    b.metric('Alignment H4/H1',f"{result['alignment']}/100")
    c.metric('Setup Readiness',f"{result['readiness']}/100")
    st.caption(f"H4 {result['state_h4']} • H1 {result['state_h1']} • M15 {result['setup']['name']} • M5 {result['trigger']['name']}")
    st.info(f"สถานะ: {result['reason']}")
    if result['plan']:
        p=result['plan']; cols=st.columns(4)
        cols[0].metric('Entry',fmt(p['entry'])); cols[1].metric('SL',fmt(p['sl'])); cols[2].metric('TP1',fmt(p['tp1'])); cols[3].metric('TP2',fmt(p['tp2']))
        st.caption(f"M15 Range: {result['range']['zone']} • R:R to TP2 ≈ {result['rr']:.2f}R")


st.title('Gold & Bitcoin Trading Analyzer — V4.3')
st.caption('D1 trend → H4/H1 range & structure → M15 setup → M5 entry trigger. Short Hold และ Long Hold แสดงพร้อมกัน')

with st.sidebar:
    st.header('ตั้งค่าการวิเคราะห์')
    asset_name=st.selectbox('สินทรัพย์',list(ASSETS.keys()),index=0)
    symbol=ASSETS[asset_name]
    chart_tf=st.selectbox('Timeframe กราฟ',list(TF.keys()),index=1)
    outputsize=st.slider('จำนวนแท่ง',250,800,500,50)
    auto=st.checkbox('Auto refresh',False)
    refresh=st.slider('รอบรีเฟรช (วินาที)',30,300,60,10)
    if st.button('รีเฟรชข้อมูลตอนนี้'): st.cache_data.clear(); st.rerun()
    st.divider(); st.caption('API: Twelve Data')

if auto:
    st.markdown(f'<meta http-equiv="refresh" content="{refresh}">',unsafe_allow_html=True)

frames={}; errors=[]
for label in ['1D','4h','1h','15m','5m']:
    raw,err=get_ohlcv(symbol,TF[label],outputsize)
    if err: errors.append(f'{label}: {err}')
    frames[label]=indicators(raw) if not raw.empty else pd.DataFrame()

chart_raw,chart_err=get_ohlcv(symbol,TF[chart_tf],outputsize)
chart=indicators(chart_raw) if not chart_raw.empty else pd.DataFrame()

if errors:
    st.error(' | '.join(errors))
    st.stop()

price=float(chart.close.iloc[-1])
prev=float(chart.close.iloc[-2]) if len(chart)>1 else price
pct=(price/prev-1)*100 if prev else 0
st.subheader(asset_name)
st.metric('Price',fmt(price),f'{pct:+.2f}%')

short=evaluate(frames,'Short Hold')
long=evaluate(frames,'Long Hold')

st.markdown('## สถานะการเทรด')
c1,c2=st.columns(2)
with c1: card(short,'Short Hold')
with c2: card(long,'Long Hold')

st.markdown('## โครงสร้างตลาด')
rows=[]
for tf in ['1D','4h','1h','15m','5m']:
    d=frames[tf]; b,s,info=trend_score(d); rg=range_info(d,48)
    rows.append({'TF':tf,'Bias':b,'Trend':s,'Structure':info['structure'],'Range':rg['zone'],'Range High':rg['high'],'Range Low':rg['low']})
st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

st.markdown(f'## กราฟ {chart_tf}')
fig=go.Figure()
fig.add_trace(go.Candlestick(x=chart.index,open=chart.open,high=chart.high,low=chart.low,close=chart.close,name='Price'))
for n in (20,50,200):
    if f'EMA{n}' in chart: fig.add_trace(go.Scatter(x=chart.index,y=chart[f'EMA{n}'],name=f'EMA{n}',mode='lines'))
fig.update_layout(height=520,xaxis_rangeslider_visible=False,margin=dict(l=10,r=10,t=30,b=10))
st.plotly_chart(fig,use_container_width=True)

st.markdown('## วิธีอ่านระบบ')
st.write('D1 ใช้กำหนดทิศทางหลักของวัน. H4 และ H1 ใช้ดูโครงสร้างและตำแหน่งราคาในช่วง. M15 ใช้หา setup เช่น pullback, breakout หรือ sweep. M5 ใช้เป็น trigger สำหรับ Short Hold และเป็นข้อมูลประกอบสำหรับ Long Hold. ระบบไม่ถือว่า Trend Score เป็นโอกาสชนะและไม่ใช้ M5 เป็น true order flow.')
