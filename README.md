# Gold & Bitcoin Trading Analyzer V2

V2 เปลี่ยนจาก Yahoo Finance เป็น Twelve Data เพื่อรองรับ XAU/USD และ BTC/USD และออกแบบให้ต่อ WebSocket ได้ในขั้นต่อไป. Twelve Data ระบุว่ามี real-time streaming ผ่าน WebSocket และครอบคลุม commodities/crypto.

## ติดตั้ง
```bash
pip install -r requirements.txt
```

ตั้ง API key:
### Windows PowerShell
```powershell
$env:TWELVEDATA_API_KEY="YOUR_KEY"
```
### macOS/Linux
```bash
export TWELVEDATA_API_KEY="YOUR_KEY"
```

รัน:
```bash
streamlit run app.py
```

## V2
- XAU/USD
- BTC/USD
- 5m / 15m / 1h / 4h / 1D
- EMA20/50/200
- RSI
- MACD
- ATR
- Support/Resistance
- BUY/SELL/WAIT score
- Entry / SL / TP
- Multi-timeframe confirmation 15m/1h/4h
- Auto refresh

## หมายเหตุ
V2 ใช้ REST polling เพื่อให้ติดตั้งง่ายและไม่ต้องจัดการ WebSocket ในเครื่องผู้ใช้. หากต้องการ "tick-by-tick" จริง ให้ต่อ WebSocket ของ data provider ใน V3.
Gold spot เป็น XAU/USD โดยตรงเมื่อ provider/account รองรับ symbol นี้; ไม่ใช่ GC=F proxy แบบ V1.
