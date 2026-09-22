Gold & Bitcoin Trading Analyzer V4.5
V4.5 is a full rebuild of the signal engine around a clear hierarchy:
D1/H4 = Macro direction -> H1/M15 = Tactical direction and setup -> M5 = entry trigger.
BTC/USD is the default asset and Short Hold + Long Hold are displayed together.
Signal states
WAIT: no usable setup yet.
PRE-ENTRY: direction/setup is developing; wait for the next trigger.
ENTRY READY: entry conditions are complete and Entry/SL/TP are shown.
Short Hold
Follows the tactical H1/M15 direction. It can identify a counter-macro tactical move, but the UI labels the macro/tactical relationship. M15 setup plus M5 trigger are required for ENTRY READY.
Long Hold
Follows the D1/H4 macro direction. If H1/M15 move against the macro direction, that is treated as a pullback/rebound rather than an automatic long/short signal. M15 setup is required; M5 helps timing.
M15 setups
Pullback, breakout/breakdown, sweep/reclaim/reject, and continuation.
Risk
Stops use recent structure plus ATR buffers. Entry/SL/TP are shown only for ENTRY READY. The analysis uses completed candles only.
M5 is candle/relative-volume confirmation, not true bid/ask delta, footprint, or DOM. Twelve Data REST polling is used; there is no automatic order placement.
Streamlit Secrets
TWELVEDATA_API_KEY = "YOUR_KEY"
