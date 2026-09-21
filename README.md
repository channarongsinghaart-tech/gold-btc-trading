Gold & Bitcoin Trading Analyzer — V4.2.2
V4.2.2 is an M30-primary analyzer with both holding horizons shown at the same time. No Short/Long mode selector is required.
Dashboard behavior
BTC/USD is first in the asset selector.
Short Hold and Long Hold are calculated and displayed together.
The user can still choose the asset and main chart timeframe.
M30 is the primary setup timeframe.
M5 is a candle/relative-volume proxy confirmation, not true footprint, bid/ask delta, or DOM.
Horizons
Short Hold
Narrower setup.
M30 setup + stronger M5 proxy confirmation.
Tighter invalidation.
TP1 = 1.25R, TP2 = 2.0R.
Long Hold
Wider structure.
D1/H4 regime with H1/M30 confirmation.
M5 can weaken the score but does not veto by itself.
Wider invalidation.
TP1 = 1.5R, TP2 = 3.0R.
Readiness and journal
Trend Score is separate from Setup Readiness.
Setup Readiness is shown even when there is no trade.
Signal IDs use the completed M30 signal bar.
Trade outcomes are evaluated only on M30 bars after the signal bar to avoid look-ahead.
Journal evaluation is restricted to the matching symbol, so BTC and Gold trades are not cross-evaluated.
If one bar touches both stop and target, the result is marked BOTH TOUCHED — REVIEW.
Data
Twelve Data REST API.
API key must be stored in Streamlit Secrets as TWELVEDATA_API_KEY.
No automatic order placement.
