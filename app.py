import os
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title='Gold & BTC Trading Analyzer V4.1 A', page_icon='📈', layout='wide')

BASE = 'https://api.twelvedata.com'
ASSETS = {'Gold XAU/USD': 'XAU/USD', 'Bitcoin BTC/USD': 'BTC/USD'}
TF = {'5m':'5min','15m':'15min','30m':'30min','1h':'1h','4h':'4h','1D':'1day'}
ANALYSIS_TFS = ['1D','4h','1h','30m','15m']

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
    d=df.copy()
    c,h,l=d.close,d.high,d.low
    for n in (20,50,200): d[f'EMA{n}']=c.ewm(span=n,adjust=False).mean()
    delta=c.diff()
    gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    rs=gain/loss.replace(0,np.nan)
    d['RSI']=100-(100/(1+rs))
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean()
    d['MACD']=e12-e26; d['MACDsig']=d.MACD.ewm(span=9,adjust=False).mean(); d['MACDhist']=d.MACD-d.MACDsig
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    d['ATR']=tr.ewm(alpha=1/14,adjust=False).mean()
    if 'volume' in d.columns:
        d['VolMA20']=d.volume.rolling(20).mean(); d['VolRatio']=d.volume/d.VolMA20
    else: d['VolMA20']=np.nan; d['VolRatio']=np.nan
    d['body']=(d.close-d.open).abs(); d['range']=d.high-d.low
    d['body_ratio']=d.body/d.range.replace(0,np.nan)
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
    if d is None or len(d)<220: return 'NEUTRAL',50, 'ข้อมูลไม่พอ'
    x=d.iloc[-1]; state=structure_state(d)
    bull=0; bear=0; reasons=[]
    bull += int(x.close>x.EMA20); bear += int(x.close<x.EMA20)
    bull += int(x.EMA20>x.EMA50); bear += int(x.EMA20<x.EMA50)
    bull += int(x.EMA50>x.EMA200); bear += int(x.EMA50<x.EMA200)
    bull += int(x.MACD>x.MACDsig); bear += int(x.MACD<x.MACDsig)
    if state in ('HH_HL','BULL'): bull+=2
    elif state in ('LH_LL','BEAR'): bear+=2
    if bull>bear: direction='LONG'; strength=int(np.clip(50+(bull-bear)*12.5,0,100))
    elif bear>bull: direction='SHORT'; strength=int(np.clip(50+(bear-bull)*12.5,0,100))
    else: direction='NEUTRAL'; strength=50
    reasons.append(state)
    reasons.append('bullish' if direction=='LONG' else 'bearish' if direction=='SHORT' else 'mixed')
    return direction,strength,'; '.join(reasons)


def setup_flags(d):
    if len(d)<30: return {}
    x=d.iloc[-1]; prev=d.iloc[-2]; prior=d.iloc[-21:-1]; atr=float(x.ATR)
    if not np.isfinite(atr) or atr<=0: return {}
    ph=float(prior.high.max()); pl=float(prior.low.min())
    bull_break=bool(x.close>ph and x.close>x.open and x.body_ratio>=0.50 and x.body>=0.35*atr)
    bear_break=bool(x.close<pl and x.close<x.open and x.body_ratio>=0.50 and x.body>=0.35*atr)
    bull_pull=bool(x.close>x.EMA20 and x.EMA20>x.EMA50 and x.low<=x.EMA20+0.15*atr and x.close>x.open and x.body_ratio>=0.35)
    bear_pull=bool(x.close<x.EMA20 and x.EMA20<x.EMA50 and x.high>=x.EMA20-0.15*atr and x.close<x.open and x.body_ratio>=0.35)
    recent_low=float(d.iloc[-8:-1].low.min()); recent_high=float(d.iloc[-8:-1].high.max())
    bull_sweep=bool(x.low<recent_low and x.close>recent_low and x.close>x.open)
    bear_sweep=bool(x.high>recent_high and x.close<recent_high and x.close<x.open)
    return {'bull_breakout':bull_break,'bear_breakdown':bear_break,'bull_pullback':bull_pull,'bear_pullback':bear_pull,'bull_sweep':bull_sweep,'bear_sweep':bear_sweep}


def levels(d, lookback=120):
    x=d.tail(lookback)
    ph=x.high.rolling(5,center=True).max(); pl=x.low.rolling(5,center=True).min()
    r=x.loc[x.high>=ph,'high'].dropna().tail(12); s=x.loc[x.low<=pl,'low'].dropna().tail(12)
    resistance=float(r.mean()) if len(r) else float(x.high.tail(30).max())
    support=float(s.mean()) if len(s) else float(x.low.tail(30).min())
    return support,resistance


def v41a_engine(frames):
    details={}
    for tf in ANALYSIS_TFS:
        d=frames.get(tf)
        if d is not None and not d.empty:
            b,s,r=bias(d); details[tf]={'Bias':b,'Strength':s,'Structure':structure_state(d),'Reason':r}
    required=ANALYSIS_TFS
    if any(tf not in details for tf in required):
        return 'NO TRADE',0,0,details,'ข้อมูล MTF ไม่ครบ'
    dirs={tf:details[tf]['Bias'] for tf in required}
    d1,h4,h1,m30,m15=[dirs[x] for x in required]
    # A = strict: D1/H4/H1 must agree; M30 cannot oppose; M15 must agree.
    if d1 not in ('LONG','SHORT') or h4!=d1 or h1!=d1 or m15!=d1:
        return 'NO TRADE',0,0,details,'Regime/entry timeframe ยังไม่ตรงกัน'
    if m30 not in (d1,'NEUTRAL'):
        return 'NO TRADE',0,0,details,'M30 สวนทางกับ trend หลัก'
    align=100 if m30==d1 else 90
    strengths=[details[t]['Strength'] for t in required if details[t]['Bias']==d1]
    score=int(round(np.mean(strengths)))
    if score<68:
        return 'NO TRADE',score,align,details,'Strength ต่ำกว่าเกณฑ์ A (68)'
    signal='STRONG LONG' if d1=='LONG' and score>=82 and align>=90 else 'STRONG SHORT' if d1=='SHORT' and score>=82 and align>=90 else d1
    return signal,score,align,details,'ผ่าน A: D1/H4/H1/M15 สอดคล้อง'


def strict_entry(frames, direction, support, resistance):
    if direction=='NO TRADE': return None
    m15=frames['15m']; h1=frames['1h']; h4=frames['4h']
    x=m15.iloc[-1]; atr=float(x.ATR); price=float(x.close)
    if not np.isfinite(atr) or atr<=0: return None
    flags=setup_flags(m15)
    if direction in ('LONG','STRONG LONG'):
        setup=flags.get('bull_pullback') or flags.get('bull_breakout') or flags.get('bull_sweep')
        if not setup: return None
        # Do not chase directly into resistance.
        if np.isfinite(resistance) and resistance>price and resistance-price<0.75*atr: return None
        invalidation=float(min(h1.tail(30).low.min(),h4.tail(20).low.min()))
        sl=invalidation-0.25*atr; risk=price-sl
        if risk<=0 or risk>3.0*atr: return None
        tp1=price+1.5*risk; tp2=price+2.5*risk
        return {'direction':'LONG','entry':price,'sl':sl,'tp1':tp1,'tp2':tp2,'risk':risk,'rr1':1.5,'rr2':2.5,'invalidation':invalidation,'setup':'Pullback/Breakout/Sweep'}
    setup=flags.get('bear_pullback') or flags.get('bear_breakdown') or flags.get('bear_sweep')
    if not setup: return None
    if np.isfinite(support) and price>support and price-support<0.75*atr: return None
    invalidation=float(max(h1.tail(30).high.max(),h4.tail(20).high.max()))
    sl=invalidation+0.25*atr; risk=sl-price
    if risk<=0 or risk>3.0*atr: return None
    tp1=price-1.5*risk; tp2=price-2.5*risk
    return {'direction':'SHORT','entry':price,'sl':sl,'tp1':tp1,'tp2':tp2,'risk':risk,'rr1':1.5,'rr2':2.5,'invalidation':invalidation,'setup':'Pullback/Breakdown/Sweep'}


def fmt(v): return f'{v:,.4f}'

st.title('Gold & Bitcoin Trading Analyzer — V4.1 A')
st.caption('โหมด A = Strict / High Selectivity. เน้นคัดกรองมากกว่า V3; สัญญาณน้อยลงได้เป็นปกติ. Confidence ไม่ใช่ win probability.')

with st.sidebar:
    st.header('ตั้งค่าการวิเคราะห์')
    asset=st.selectbox('สินทรัพย์',list(ASSETS.keys()))
    timeframe=st.selectbox('Timeframe หลัก',list(TF.keys()),index=1)
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
d=add_indicators(main_df); last=d.iloc[-1]; price=float(last.close); prev=float(d.close.iloc[-2]); pct=(price/prev-1)*100
support,resistance=levels(d)

frames={}; errors={}
for tf in ANALYSIS_TFS:
    x,er=get_ohlcv(symbol,TF[tf],outputsize)
    if not er and len(x)>=220: frames[tf]=add_indicators(x)
    else: errors[tf]=er or 'ข้อมูลไม่พอ'

vsignal,vscore,alignment,details,gate_reason=v41a_engine(frames)
risk=strict_entry(frames,vsignal,support,resistance) if vsignal!='NO TRADE' else None
confidence=int(np.clip(0.65*vscore+0.35*alignment,50,99)) if vsignal!='NO TRADE' else 0

c1,c2,c3,c4,c5=st.columns(5)
c1.metric('Price',fmt(price),f'{pct:+.2f}%')
c2.metric('V4.1 A Signal',vsignal)
c3.metric('Score',f'{vscore}/100')
c4.metric('Confidence',f'{confidence}%' if confidence else '—')
c5.metric('RSI',f'{last.RSI:.1f}')
st.caption(f'แท่งล่าสุด: {last.name.strftime("%Y-%m-%d %H:%M UTC")} • {symbol} • {timeframe}')

fig=go.Figure(); fig.add_trace(go.Candlestick(x=d.index,open=d.open,high=d.high,low=d.low,close=d.close,name=symbol))
for col in ['EMA20','EMA50','EMA200']: fig.add_trace(go.Scatter(x=d.index,y=d[col],name=col,mode='lines'))
fig.add_hline(y=support,annotation_text='Support',line_dash='dot'); fig.add_hline(y=resistance,annotation_text='Resistance',line_dash='dot')
fig.update_layout(height=620,xaxis_rangeslider_visible=False,margin=dict(l=10,r=10,t=30,b=10)); st.plotly_chart(fig,use_container_width=True)

left,right=st.columns(2)
with left:
    st.subheader('V4.1 A Gate')
    st.write(f'**{gate_reason}**')
    st.write(f'Alignment: **{alignment}%**')
    st.write(f'Support: **{fmt(support)}**')
    st.write(f'Resistance: **{fmt(resistance)}**')
with right:
    st.subheader('Entry / SL / TP')
    if risk:
        st.write(f"Direction: **{risk['direction']}**")
        st.write(f"Entry: **{fmt(risk['entry'])}**")
        st.write(f"Stop Loss: **{fmt(risk['sl'])}**")
        st.write(f"TP1: **{fmt(risk['tp1'])}** (R:R 1:{risk['rr1']:.1f})")
        st.write(f"TP2: **{fmt(risk['tp2'])}** (R:R 1:{risk['rr2']:.1f})")
        st.caption(f"Risk/Unit: {fmt(risk['risk'])} • Invalidation: {fmt(risk['invalidation'])}")
    elif vsignal!='NO TRADE': st.warning('Trend ผ่าน แต่ Entry Filter A ยังไม่ผ่าน — รอจังหวะ')
    else: st.write('NO TRADE — ไม่ผ่าน Strict Gate')

st.subheader('Top-Down Multi-Timeframe')
rows=[]
for tf in ANALYSIS_TFS:
    if tf in details:
        z=details[tf]; rows.append([tf,z['Bias'],z['Strength'],z['Structure'],z['Reason']])
    else: rows.append([tf,'ERROR','—','—',errors.get(tf,'ไม่มีข้อมูล')])
st.dataframe(pd.DataFrame(rows,columns=['Timeframe','Bias','Strength','Structure','Reason']),hide_index=True,use_container_width=True)

st.subheader('Current Setup')
flags=setup_flags(d); active=[]
labels={'bull_breakout':'Bullish Breakout','bear_breakdown':'Bearish Breakdown','bull_pullback':'Bullish Pullback','bear_pullback':'Bearish Pullback','bull_sweep':'Bullish Liquidity Sweep','bear_sweep':'Bearish Liquidity Sweep'}
for k,v in flags.items():
    if v: active.append(labels[k])
st.write(' • '.join(active) if active else 'ยังไม่พบ setup ที่ผ่านโครงสร้างล่าสุด')

st.subheader('Indicator Snapshot')
st.dataframe(pd.DataFrame([
    ['EMA20',float(last.EMA20)],['EMA50',float(last.EMA50)],['EMA200',float(last.EMA200)],
    ['RSI14',float(last.RSI)],['MACD',float(last.MACD)],['MACD Signal',float(last.MACDsig)],['ATR14',float(last.ATR)]
],columns=['Indicator','Value']),hide_index=True,use_container_width=True)

st.caption('V4.1 A เป็นระบบวิเคราะห์ ไม่ส่งคำสั่งซื้อขายอัตโนมัติ. Strict mode อาจแสดง NO TRADE เป็นส่วนใหญ่เมื่อ regime ไม่ชัดหรือ entry อยู่ใกล้แนวรับ/แนวต้าน. ความสดของข้อมูลขึ้นกับแพ็กเกจ Twelve Data และ API credits.')
