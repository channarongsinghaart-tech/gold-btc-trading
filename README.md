Gold & Bitcoin Trading Analyzer — V4.2.2
V4.2.2 keeps the M30-primary two-horizon design and makes the journal look-ahead-safe.
Core structure
D1 → H4 → H1 → M30 setup → M5 proxy confirmation
Modes
Short Hold: narrower setup, stronger M5 confirmation, tighter risk, TP1 1.25R / TP2 2R
Long Hold: wider setup, H1/H4 + M30 structure, M5 is confirmatory, TP1 1.5R / TP2 3R
V4.2.2 changes
Uses completed candles for signal generation; the newest forming candle is excluded from analysis.
Replaces Entry Quality display with Setup Readiness to distinguish setup readiness from trend strength.
Shows Setup Readiness even when the signal is NO TRADE.
Signal ID includes the completed M30 signal bar.
Journal evaluates outcomes only on M30 bars strictly after the signal bar, preventing look-ahead from the signal candle.
If SL and TP1/TP2 are both touched in the same M30 bar, status becomes BOTH TOUCHED — REVIEW because intrabar order is unknown.
Existing trades continue to be evaluated even when the current setup changes to NO TRADE.
Journal includes Signal ID and Signal Bar.
Data limitations
M5 is a candle/relative-volume proxy. It is not true bid/ask delta, footprint, DOM, or centralized futures order-flow data.
The app uses Twelve Data REST polling and does not place orders automatically.
Streamlit Secrets
Set:
TWELVEDATA_API_KEY = "your_private_key"
Do not put the key in GitHub source code.
