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
