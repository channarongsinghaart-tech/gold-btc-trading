Gold & Bitcoin Trading Analyzer V4.3
โครงสร้างใหม่แบบพื้นฐานและโปร่งใส:
BTC/USD แสดงเป็นสินทรัพย์เริ่มต้น
Short Hold และ Long Hold แสดงพร้อมกัน ไม่ต้องเลือกโหมด
D1 = ทิศทางหลักของวัน
H4/H1 = trend + structure + range ของกราฟ
M15 = หา setup: pullback / breakout / sweep
M5 = หา entry trigger สำหรับ Short Hold; Long Hold ใช้เป็นข้อมูลประกอบ ไม่บังคับ
Entry / SL / TP ใช้ M15 + โครงสร้าง H1/H4
ไม่มี true footprint / bid-ask delta / DOM
ใช้ Twelve Data REST API
ไม่ส่งคำสั่งซื้อขายอัตโนมัติ
ต้องตั้ง Streamlit Secret: TWELVEDATA_API_KEY = "YOUR_KEY"
