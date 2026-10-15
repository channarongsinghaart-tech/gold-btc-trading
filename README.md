Gold & Bitcoin Trading Analyzer — Trend Following + LINE Push
ชุดนี้แยกเป็น 2 ส่วน:
app.py — Streamlit analyzer สำหรับดูกราฟ/สถานะ Short Hold และ Long Hold ใช้ Trend Following เท่านั้น ไม่มี Counter-Macro Entry
background_scanner.py + GitHub Actions — ตรวจ BTC และ Gold เป็นระยะและส่ง LINE Push ไปยัง LINE_USER_ID ที่กำหนด แม้ไม่ได้เปิด Streamlit app
GitHub Secrets ที่ต้องตั้ง Repository → Settings → Secrets and variables → Actions → New repository secret:
TWELVEDATA_API_KEY
LINE_CHANNEL_ACCESS_TOKEN
LINE_USER_ID ห้ามใส่ secret ลงใน source code
Streamlit Secrets ใน Streamlit Cloud → Settings → Secrets ใช้:
TWELVEDATA_API_KEY = "..."
LINE_CHANNEL_ACCESS_TOKEN = "..."
LINE_USER_ID = "U..."
โครงสร้างการตัดสินใจ (อัปเดตล่าสุด)
ใช้แค่ 4 timeframe: D1, H4, H1, M15 — ไม่มี M5 หรือ M30 trigger แยกแล้ว
D1/H4 = Macro, H1/M15 = Tactical
M15 ทำหน้าที่ทั้ง setup และจุดเข้าในตัวเดียว (pullback/sweep/breakout/continuation) ไม่ต้องรอ trigger จาก timeframe ที่เร็วกว่าอีกขั้น
ต้อง D1/H4/H1/M15 bias ตรงกันทั้งหมด (full align) และ Macro Strength ≥ 60 ก่อนถึงจะ ENTRY READY ได้
Short Hold และ Long Hold ยังต่างกันด้วยช่วง SL (swing lookback) และตัวคูณ TP (1.2R/1.8R กับ 1.5R/3.0R) แม้จะใช้ M15 เป็นจุดเข้าเดียวกันก็ตาม
Scanner
GitHub Actions รันทุก 15 นาที เฉพาะช่วง 4:00 AM–9:30 PM เวลาอัลเบอร์ตา (America/Edmonton, ปรับ DST อัตโนมัติ) และสามารถกด Run workflow เองได้
โหลดข้อมูลแค่ 4 requests ต่อรอบสแกน (D1/H4/H1/M15) ต่อสินทรัพย์ ลดลงจากเดิมที่เคยมี 5 timeframe
มี cooldown 45 นาทีต่อ (สินทรัพย์+โหมด+ทิศทาง) กันแจ้งเตือนทิศเดียวกันซ้ำถี่เกินไป เก็บสถานะไว้ใน alert_state.json
หมายเหตุ: GitHub Actions แบบ schedule อาจมีความล่าช้าเล็กน้อยจากระบบ GitHub และไม่ใช่ real-time tick alert
