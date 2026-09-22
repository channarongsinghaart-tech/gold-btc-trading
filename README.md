Gold & Bitcoin Trading Analyzer — V4.7
V4.7 fixes signal classification and the Counter-Macro handling from V4.6 without changing the Twelve Data/Secrets setup.
Signal hierarchy
D1 + H4 = Macro direction
H1 + M15 = Tactical direction
M15 = setup
M5 = entry trigger
Signal classes
TREND ENTRY: Macro and Tactical point in the same direction.
COUNTER-MACRO ENTRY: Short Hold only; Tactical points against Macro. The UI explicitly labels this as Counter-Macro instead of presenting it as a normal trend entry.
PULLBACK / RESUME: Long Hold while Tactical is still opposite Macro. Long Hold cannot show ENTRY READY in this state; it waits for H1/M15 to realign with Macro.
WAIT: No valid direction/setup/trigger.
Important behavior change
V4.6 could display Long Hold ENTRY READY while Tactical H1/M15 was still opposite Macro. V4.7 removes that condition.
Short Hold can still produce a Counter-Macro entry when M15 setup + M5 trigger are both confirmed. This is deliberately classified separately from a Trend Entry.
Setup Readiness is a setup-quality score, not a win probability.
The scanner checks the latest 3 closed candles for M15 setup and M5 trigger so a valid event is less likely to be missed between refreshes.
Entry/SL/TP are shown only for ENTRY READY.
M5 is candle/relative-volume confirmation only. It is not true bid/ask delta, footprint, DOM, or centralized futures order-flow data.
Data source: Twelve Data REST. The app does not place orders automatically.
