Gold & Bitcoin Trading Analyzer V4.7 — Trend Following
ChaGold & Bitcoin Trading Analyzer V4.7 — LINE Push
V4.7 Trend-following analyzer using Twelve Data.
LINE Push
The app uses the LINE Messaging API push endpoint and sends only to the exact LINE_USER_ID stored in Streamlit Secrets. It does not use broadcast.
Add these to Streamlit Secrets:
TWELVEDATA_API_KEY = ""
LINE_CHANNEL_ACCESS_TOKEN = "2eaa3e7096d9f4035d674ee0683f320d"
LINE_USER_ID = "U3d78e626c70085bcd9da189d9f77abfc"
The app sends an alert only when Short Hold or Long Hold Gold & Bitcoin Trading Analyzer V4.7 — Trend Following
ChaGold & Bitcoin Trading Analyzer V4.7 — LINE Push
V4.7 Trend-following analyzer using Twelve Data.
LINE Push
The app uses the LINE Messaging API push endpoint and sends only to the exact LINE_USER_ID stored in Streamlit Secrets. It does not use broadcast.
Add these to Streamlit Secrets:
TWELVEDATA_API_KEY = "..."
LINE_CHANNEL_ACCESS_TOKEN = "..."
LINE_USER_ID = "U..."
The app sends an alert only when Short Hold or Long Hold becomes ENTRY READY and a complete Entry / SL / TP1 / TP2 plan exists. It suppresses duplicate alerts for the same setup during the active Streamlit session.
A ทดสอบ LINE Push button is available in the sidebar and also targets only the configured LINE_USER_ID.nges in this build:
Short Hold is trend-following only; no COUNTER-MACRO ENTRY.
ENTRY READY requires D1/H4/H1/M15 alignment.
Macro Strength must be at least 60/100.
M5 trigger score uses only breakout, reclaim and rejection events.
Volume and EMA-position points are not added to M5 decision score.
Chart BUY/SELL markers remain view-only raw candle markers.
Twelve Data API key is read from TWELVEDATA_API_KEY Streamlit Secret.
No order execution is included.
 ENTRY READY and a complete Entry / SL / TP1 / TP2 plan exists. It suppresses duplicate alerts for the same setup during the active Streamlit session.
A ทดสอบ LINE Push button is available in the sidebar and also targets only the configured LINE_USER_ID.nges in this build:
Short Hold is trend-following only; no COUNTER-MACRO ENTRY.
ENTRY READY requires D1/H4/H1/M15 alignment.
Macro Strength must be at least 60/100.
M5 trigger score uses only breakout, reclaim and rejection events.
Volume and EMA-position points are not added to M5 decision score.
Chart BUY/SELL markers remain view-only raw candle markers.
Twelve Data API key is read from TWELVEDATA_API_KEY Streamlit Secret.
No order execution is included.
