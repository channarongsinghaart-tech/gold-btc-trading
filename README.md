Gold & Bitcoin Trading Analyzer V4.2.1
V4.2.1 is a mobile-friendly Streamlit analyzer using Twelve Data.
Core structure
D1 -> H4 -> H1 -> M30
M30 is the primary setup timeframe.
M5 is confirmation only and is a proxy based on candle/relative-volume behavior, not true bid/ask delta, footprint, DOM, or Level 2 order flow.
Two holding modes
Short Hold: narrower setup, stronger M5 confirmation, tighter structure-based risk, TP1 1.25R / TP2 2.0R.
Long Hold: wider setup, D1/H4 regime plus H1/M30 confirmation, M5 does not veto by itself, TP1 1.5R / TP2 3.0R.
New in V4.2.1
Entry Quality separated from Trend Score.
Detailed NO TRADE / WAIT reasons.
Regime-aware Entry Quality component.
Signal cooldown: one active setup per Symbol + Mode + Direction to avoid duplicate signals on refresh.
Session Trade Journal with Entry/SL/TP1/TP2 and R-based results.
TP1 is treated as a partial milestone; TP1 + SL assumes 50% taken at TP1 and 50% stopped.
Profit Factor, Win Rate, Average R, Max Drawdown R, and open-trade count.
CSV export and session reset button.
Important
The journal is session-only. It resets when the Streamlit session restarts.
If TP1 and TP2/SL are touched in the same M30 bar, the trade is marked BOTH TOUCHED — REVIEW because candle OHLC cannot establish intrabar order.
The analyzer does not place orders automatically.
Score and Entry Quality are not win probabilities.
Secrets
Set TWELVEDATA_API_KEY in Streamlit Secrets. Do not commit the API key to GitHub.
