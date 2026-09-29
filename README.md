Gold & Bitcoin Trading Analyzer V4.7 — LINE Push
V4.7 Trend-following analyzer using Twelve Data.
LINE Push
The app uses the LINE Messaging API push endpoint and sends only to the exact LINE_USER_ID stored in Streamlit Secrets. It does not use broadcast.
Add these to Streamlit Secrets:
TWELVEDATA_API_KEY = "..."
LINE_CHANNEL_ACCESS_TOKEN = "..."
LINE_USER_ID = "U..."
The app sends an alert only when Short Hold or Long Hold becomes ENTRY READY and a complete Entry / SL / TP1 / TP2 plan exists. It suppresses duplicate alerts for the same setup during the active Streamlit session.
A ทดสอบ LINE Push button is available in the sidebar and also targets only the configured LINE_USER_ID.
