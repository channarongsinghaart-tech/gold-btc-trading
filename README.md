Gold & Bitcoin Trading Analyzer — Trend Following + LINE Push
ชุดนี้แยกเป็น 2 ส่วน:
app.py — Streamlit analyzer สำหรับดูกราฟ/สถานะ Short Hold และ Long Hold. ใช้ Trend Following เท่านั้น และไม่มี Counter-Macro Entry.
background_scanner.py + GitHub Actions — ตรวจ BTC และ Gold เป็นระยะและส่ง LINE Push ไปยัง LINE_USER_ID ที่กำหนด แม้ไม่ได้เปิด Streamlit app.
GitHub Secrets ที่ต้องตั้ง
Repository → Settings → Secrets and variables → Actions → New repository secret:
TWELVEDATA_API_KEY
LINE_CHANNEL_ACCESS_TOKEN
LINE_USER_ID
ห้ามใส่ secret ลงใน source code.
Streamlit Secrets
ใน Streamlit Cloud → Settings → Secrets ใช้:
TWELVEDATA_API_KEY = "..."
LINE_CHANNEL_ACCESS_TOKEN = "..."
LINE_USER_ID = "U..."
Scanner
GitHub Actions รันทุก 10 นาทีและสามารถกด Run workflow เองได้. Scanner ตรวจ D1/H4/H1/M15 ให้ไปทางเดียวกัน, Macro Strength ≥ 60, จากนั้นต้องมี M15 setup และ M5 trigger ก่อนส่ง ENTRY READY.
หมายเหตุ: GitHub Actions แบบ schedule อาจมีความล่าช้าเล็กน้อยจากระบบ GitHub และไม่ใช่ real-time tick alert.
