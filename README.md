# ASL-XYZ (A–Y realtime, custom XYZ features)

ระบบรู้จำตัวอักษรภาษาใบ้ A–Y (ตัด J,Z) แบบเรียลไทม์ บน Windows และ Raspberry Pi  
- กล้องเว็บแคม + MediaPipe  
- ฟีเจอร์คำนวณเองจาก XYZ: ระยะ, มุม (PIP/DIP/IP), เวกเตอร์, dot/cross, pairwise tip distances, palm orientation (pitch/yaw/roll)  
- 2 โหมด: Heuristic (ไม่ต้องฝึก) และ ML (kNN/SVM) ฝึกจากข้อมูลผู้ใช้

> แรงบันดาลใจ/แนวคิดฟีเจอร์สัมพันธ์มือ–ร่างกาย (ST-BHR) จากงานวิจัย: Sensors 2022 22:4554. :contentReference[oaicite:1]{index=1}

## Quick start
```bash
# 1) clone
git clone <your-repo-url> && cd asl-xyz

# 2) env
python -m venv .venv
# (Windows) .venv\Scripts\Activate.ps1
# (Linux/RPi) source .venv/bin/activate
pip install -r requirements.txt

# 3) run
python src/asl_realtime.py
