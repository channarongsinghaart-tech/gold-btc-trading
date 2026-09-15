import os
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title='Gold & BTC Trading Analyzer V4.2', page_icon='📈', layout='wide')

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
    details={}
    for tf in ANALYSIS_TFS:
        d=frames.get(tf)
        if d is not None and not d.empty:
            b,s,r=bias(d); env,envs,envr=environment(d)
            details[tf]={'Bias':b,'Strength':s,'Structure':structure_state(d),'Environment':env,'EnvScore':envs,'Reason':r}
    if any(tf not in details for tf in ANALYSIS_TFS): return 'NO TRADE',0,0,details,'MTF data incomplete'
    dirs={tf:details[tf]['Bias'] for tf in ANALYSIS_TFS}; d1,h4,h1,m30=[dirs[x] for x in ANALYSIS_TFS]
    if d1 not in ('LONG','SHORT') or h4!=d1: return 'NO TRADE',0,0,details,'D1/H4 regime not aligned'
    confirmations=sum(x==d1 for x in (h1,m30)); opposed=sum(x not in (d1,'NEUTRAL') for x in (h1,m30))
    strengths=[details[t]['Strength'] for t in ANALYSIS_TFS if details[t]['Bias']==d1]
    trend_score=int(round(np.mean(strengths))) if strengths else 50
    align=int(np.clip(40+20+13.33*confirmations,0,100))
    if mode=='A — Strict' and (confirmations<3 or opposed>0 or trend_score<68):
        return 'NO TRADE',trend_score,align,details,'A Strict: MTF/strength gate not passed'
    if mode=='B — Balanced' and (confirmations<1 or opposed>=2 or trend_score<55):
        return 'NO TRADE',trend_score,align,details,'B Balanced: regime/confirmation weak'
    flow_dir='LONG' if flow['score']>=68 else 'SHORT' if flow['score']<=32 else 'NEUTRAL'
    flow_bonus=abs(flow['score']-50)*0.35
    final_score=int(np.clip(round(.65*trend_score+.20*align+.15*flow['score']),0,100))
    # Do not let proxy flow override D1/H4 regime. It only confirms or weakens the setup.
    if flow_dir not in ('NEUTRAL',d1):
        final_score=max(0,final_score-12)
        flow_note='proxy order flow conflicts with regime'
    else:
        flow_note='proxy order flow supportive/neutral'
    if mode=='A — Strict' and flow_dir not in (d1,'NEUTRAL'):
        return 'NO TRADE',final_score,align,details,'A Strict: order-flow proxy conflicts with trend'
    strong=final_score>=78 and (confirmations>=2 if mode=='B — Balanced' else confirmations==2)
    signal=('STRONG LONG' if d1=='LONG' else 'STRONG SHORT') if strong else d1
    return signal,final_score,align,details,f'{mode}: {flow_note}'


def entry_plan(frames, direction, support, resistance, mode, flow, loc, vp):
    if direction=='NO TRADE': return None,'No directional signal'
    base='LONG' if 'LONG' in direction else 'SHORT'; m30=frames.get('30m'); h1=frames.get('1h'); h4=frames.get('4h')
    if any(x is None or x.empty for x in (m30,h1,h4)): return None,'MTF entry data incomplete'
    x=m30.iloc[-1]; atr=float(x.ATR); price=float(x.close)
    if not np.isfinite(atr) or atr<=0: return None,'ATR invalid'
    flags=setup_flags(m30)
    location_ok=False; location_reason=''
    if vp:
        if base=='LONG': location_ok=vp.get('location') in ('Below value / discount','Inside value')
        else: location_ok=vp.get('location') in ('Above value / premium','Inside value')
        location_reason=vp.get('location','')
    if base=='LONG':
        setup=flags.get('bull_pullback') or flags.get('bull_breakout') or flags.get('bull_sweep')
        near_ema=np.isfinite(x.EMA20) and abs(price-float(x.EMA20))<=(.60 if mode=='B — Balanced' else .30)*atr and price>float(x.EMA50)
        flow_ok=flow['score']>=55
        fib_ok=(loc.get('label','').startswith('Bullish deep')) or location_ok or near_ema
        if not (setup or near_ema): return None,'Entry: no pullback/breakout/sweep or EMA20 proximity'
        if mode=='A — Strict' and not (flow_ok and fib_ok): return None,'A Strict: flow/location confirmation missing'
        if mode=='B — Balanced' and not (flow_ok and (fib_ok or setup)): return None,'B Balanced: confirmation still weak'
        if np.isfinite(resistance) and resistance>price and resistance-price<(.65 if mode=='A — Strict' else .30)*atr: return None,'Too close to resistance'
        invalidation=float(min(h1.tail(30).low.min(),h4.tail(20).low.min(),m30.tail(12).low.min()))
        sl=invalidation-(.25 if mode=='A — Strict' else .20)*atr; risk=price-sl
        max_risk=3.0 if mode=='A — Strict' else 4.0
        if risk<=0 or risk>max_risk*atr: return None,'Risk distance too large'
        tp1=price+1.5*risk; tp2=price+2.5*risk
        return {'direction':'LONG','entry':price,'sl':sl,'tp1':tp1,'tp2':tp2,'risk':risk,'rr1':1.5,'rr2':2.5,'invalidation':invalidation,'setup':'Pullback/Breakout/Sweep' if setup else 'EMA20 proximity','flow_score':flow['score'],'location':location_reason},'READY'
    setup=flags.get('bear_pullback') or flags.get('bear_breakdown') or flags.get('bear_sweep')
    near_ema=np.isfinite(x.EMA20) and abs(price-float(x.EMA20))<=(.60 if mode=='B — Balanced' else .30)*atr and price<float(x.EMA50)
    flow_ok=flow['score']<=45
    fib_ok=(loc.get('label','').startswith('Premium')) or location_ok or near_ema
    if not (setup or near_ema): return None,'Entry: no pullback/breakdown/sweep or EMA20 proximity'
    if mode=='A — Strict' and not (flow_ok and fib_ok): return None,'A Strict: flow/location confirmation missing'
    if mode=='B — Balanced' and not (flow_ok and (fib_ok or setup)): return None,'B Balanced: confirmation still weak'
    if np.isfinite(support) and price>support and price-support<(.65 if mode=='A — Strict' else .30)*atr: return None,'Too close to support'
    invalidation=float(max(h1.tail(30).high.max(),h4.tail(20).high.max(),m30.tail(12).high.max()))
    sl=invalidation+(.25 if mode=='A — Strict' else .20)*atr; risk=sl-price
    max_risk=3.0 if mode=='A — Strict' else 4.0
    if risk<=0 or risk>max_risk*atr: return None,'Risk distance too large'
    tp1=price-1.5*risk; tp2=price-2.5*risk
    return {'direction':'SHORT','entry':price,'sl':sl,'tp1':tp1,'tp2':tp2,'risk':risk,'rr1':1.5,'rr2':2.5,'invalidation':invalidation,'setup':'Pullback/Breakdown/Sweep' if setup else 'EMA20 proximity','flow_score':flow['score'],'location':location_reason},'READY'


def fmt(v): return f'{v:,.4f}'

st.title('Gold & Bitcoin Trading Analyzer — V4.2')
st.caption('V4.2 M30-Primary = D1 → H4 → H1 → M30 setup → M5 confirmation. Order-flow is a proxy, not true footprint/delta/DOM.')

with st.sidebar:
    st.header('ตั้งค่าการวิเคราะห์')
    mode=st.radio('โหมดสัญญาณ',['A — Strict','B — Balanced'],index=0,help='A คัดเข้ม; B เปิดโอกาสมากขึ้น')
    st.info('โครงสร้าง V4.2: D1 → H4 → H1 → M30 (หลัก) → M5 (ยืนยัน)')
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
d=add_indicators(main_df); last=d.iloc[-1]; price=float(last.close); prev=float(d.close.iloc[-2]); pct=(price/prev-1)*100
support,resistance=levels(d)

frames={}; errors={}
for tf in ANALYSIS_TFS:
    x,er=get_ohlcv(symbol,TF[tf],outputsize)
    if not er and len(x)>=220: frames[tf]=add_indicators(x)
    else: errors[tf]=er or 'ข้อมูลไม่พอ'

# 30m is the primary setup timeframe; 5m is used only as the lowest-timeframe confirmation. This is deliberately called a proxy, not true order flow.
five_df,five_err=get_ohlcv(symbol,'5min',max(250,min(800,outputsize)))
five=add_indicators(five_df) if not five_err and len(five_df)>=40 else pd.DataFrame()
flow=proxy_orderflow(five)
loc=fib_location(five if not five.empty else d)
vp=volume_profile_proxy(five if not five.empty else d)

env,env_score,env_reason=environment(frames.get('4h',d))
vsignal,vscore,alignment,details,gate_reason=v42_engine(frames,mode,flow)
risk,entry_reason=entry_plan(frames,vsignal,support,resistance,mode,flow,loc,vp) if vsignal!='NO TRADE' else (None,'NO TRADE')
confidence=int(np.clip(.55*vscore+.25*alignment+.20*abs(flow['score']-50)*2,50,99)) if vsignal!='NO TRADE' else 0

c1,c2,c3,c4,c5=st.columns(5)
c1.metric('Price',fmt(price),f'{pct:+.2f}%')
c2.metric('V4.2 Signal',vsignal)
c3.metric('Score',f'{vscore}/100')
c4.metric('Confidence',f'{confidence}%' if confidence else '—')
c5.metric('RSI',f'{last.RSI:.1f}')
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
else:
    st.warning(f'WAIT / NO TRADE — {entry_reason}')

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

st.caption('V4.2 ไม่ส่งคำสั่งซื้อขายอัตโนมัติ. ใช้ Twelve Data REST; 5m order-flow เป็น proxy เพราะ Twelve Data ยังไม่มี Level 1/2/3 หรือ order book. Score/Confidence ไม่ใช่ win probability.')
