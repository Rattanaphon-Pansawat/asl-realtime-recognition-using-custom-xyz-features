# ASL Realtime Recognition using Custom XYZ Features

ระบบรู้จำตัวอักษร ASL A–Y แบบเรียลไทม์ (ตัด J,Z) บน Windows และ Raspberry Pi  
ใช้ MediaPipe landmarks และ “ฟีเจอร์ที่สร้างเองทั้งหมด” จาก XYZ (มุม ระยะ เวกเตอร์ ฯลฯ)  
รองรับ 2 โหมด: Heuristic (ไม่ต้องฝึก) และ ML (k-NN)

## วิธีติดตั้ง
```bash
# 1) Clone repo
git clone https://github.com/Rattanaphon-Pansawat/asl-realtime-recognition-using-custom-xyz-features.git
cd asl-realtime-recognition-using-custom-xyz-features

# 2) สร้าง virtual environment
python -m venv .venv
# (Windows) .venv\Scripts\Activate.ps1
# (Linux/RPi) source .venv/bin/activate

# 3) ติดตั้ง dependencies
pip install -r requirements.txt

# 4) รันโปรแกรม
python src/asl_realtime.py



## วิธีการใช้งาน

เมื่อรันโปรแกรม ระบบจะเปิดกล้องและแสดงผลการทำนายตัวอักษร ASL (A–Y)

### UI
- **Pred**: แสดงตัวอักษรที่ทำนายได้ เช่น "Pred: A"
- **Conf**: ความมั่นใจของการทำนาย เช่น "conf=0.99"
- **Mode**: โหมดที่ใช้งาน (Heuristic หรือ ML) เช่น "mode=Heuristic"

### ปุ่มลัดที่ใช้ในโปรแกรม:
- **H**: สลับโหมดระหว่าง Heuristic และ ML
- **[` `]**: เปลี่ยนตัวอักษรปัจจุบันจาก A–Y (ไม่รวม J,Z)
- **C**: เก็บฟีเจอร์จากการทำนาย
- **T**: ฝึกโมเดลจากข้อมูลที่เก็บ
- **E**: ประเมินผลการฝึกด้วย k-fold cross-validation
- **S**: บันทึกข้อมูลตัวอย่างลงในไฟล์ CSV
- **L**: โหลดข้อมูลที่บันทึกไว้ก่อนหน้า
- **Q**: ออกจากโปรแกรม


---

## ขั้นตอนที่ 3: การเทรนและการจำโมเดล
### 4.1 หลังจากการเทรน
- เมื่อทำการเทรน (กด `T`), โมเดลจะถูกฝึกบนข้อมูลที่เก็บไว้ และสามารถนำไปใช้ในการทำนายได้
- ข้อมูลถูกบันทึกในไฟล์ CSV ที่สามารถโหลดกลับมาได้ (`S` เพื่อเซฟข้อมูล)

### 4.2 จำโมเดลในครั้งถัดไป
- หากต้องการให้โปรแกรมจำการฝึกที่แล้วได้ **จำเป็นต้องบันทึกโมเดล**:
  - แนะนำให้ **บันทึกโมเดล** ด้วย `joblib` หรือ `pickle` ในไฟล์เช่น `model.pkl` หรือ `knn_model.joblib` เพื่อให้สามารถโหลดกลับมาใช้ในครั้งถัดไป

```python
import joblib
# บันทึกโมเดล
joblib.dump(knn_classifier, 'knn_model.joblib')

# โหลดโมเดล
knn_classifier = joblib.load('knn_model.joblib')

