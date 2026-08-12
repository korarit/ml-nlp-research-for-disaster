# ระบบประมวลผลข้อความภัยพิบัติด้วย Classical ML & BiLSTM-CRF (Hybrid GPU/CPU Architecture)

ระบบวิจัยและประมวลผลข้อความขอความช่วยเหลือภัยพิบัติภาษาไทยจากโซเชียลมีเดีย โดยใช้สถาปัตยกรรม **Classical Machine Learning, Rule-Based Engine และ BiLSTM-CRF** ปราศจากความยึดติดกับ API ของ Large Language Models (LLMs) เพื่อความเร็วในการทำนายระดับมิลลิวินาทีและความน่าเชื่อถือสูงสุด

---

## 📌 คุณสมบัติหลัก (Core Features)

1. **Task 1: Disaster Tweet Classification**
   - จำแนกข้อความขอความช่วยเหลือฉุกเฉิน (`help_request`) vs ข้อความทั่วไป (`other`)
   - รองรับการประเมิน 17 Classical ML Classifiers + Baselines ร่วมกับ Optuna Hyperparameter Auto-Tuning บน 5-Fold Stratified Cross-Validation
2. **Task 2: NER & Entity / Count Extraction**
   - สกัดข้อมูลเฉพาะ (เบอร์โทรศัพท์, พิกัด Lat/Lng, Google Maps URL) และทำนายจำนวนนับ (`gt_dead`, `gt_critical`, `gt_urgent`, `gt_food`, ฯลฯ)
   - เปรียบเทียบ 5 Approaches: Rule-Based, Binned Classifiers, Continuous Regressors, CRF Sequence Tagger และ Hybrid System ⭐
3. **Task 3: People Extraction & Clinical Triage Classification**
   - **Sub-task 3.1:** สกัดเหยื่อรายบุคคล (Individual Victims) และอาการป่วยด้วย Clause Splitter Rules & BiLSTM-CRF
   - **Sub-task 3.2:** Pediatric Clinical Triage Classification (เด็ก $\le 12$ ปี) ตามเกณฑ์ Pediatric IITT
   - **Sub-task 3.3:** Adult Clinical Triage Classification (ผู้ใหญ่ $> 12$ ปี) ตามเกณฑ์ Adult IITT
   - วัดผลความปลอดภัยด้วย **Critical Under-Triage Rate (UR) พร้อม 95% Confidence Interval** (Bootstrapping) และ Quadratic Weighted Kappa (QWK)
4. **Task 4: End-to-End Integrated Streaming Pipeline**
   - ร้อยต่อการทำงานทุกโมเดลเข้าด้วยกันเป็นระบบ End-to-End แบบสตรีมมิ่งในข้อความเดียว
5. **Rigorous Latency Protocol & Statistical Significance Testing**
   - วัดเวลาประมวลผลจริง (Empirical Latency) ทั้งบน GPU และ CPU (Mean, p50, p95, p99, QPS)
   - ทดสอบนัยสำคัญทางสถิติ McNemar's Test + Holm-Bonferroni Correction และ Wilcoxon Signed-Rank Test บน Residuals
6. **Discord Webhook Notifications Integration**
   - แจ้งเตือนความก้าวหน้าการเทรนและสรุปผลประสิทธิภาพเข้า Discord สดๆ ระหว่างรัน

---

## 🛠️ การติดตั้งและเตรียมความพร้อม (Installation & Setup)

### 1. ติดตั้ง Dependencies
```bash
pip install torch numpy pandas scikit-learn pythainlp optuna matplotlib seaborn xgboost lightgbm catboost python-dotenv requests
```

### 2. ตั้งค่า Discord Webhook (Optional)
สร้างไฟล์ `.env` ใน Root Directory สำหรับรับการแจ้งเตือนสด:
```env
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your_webhook_url_here
```

---

## 🚀 วิธีการสั่งรันการทดลอง (Execution Guide)

สคริปต์หลักสำหรับควบคุมระบบคือ `run_classical_ml.py`

### 🔹 1. รันการทดลองหลัก Task 1 (Disaster Tweet Classification)
รันทุกโมเดล (17 Classifiers + Baselines) ด้วย 5-Fold Stratified CV และจูน Optuna 10 รอบต่อโมเดล:
```bash
python run_classical_ml.py --task 1 --model all --use_gpu true --run_id run_02_autotune_hyperparams --n_trials 10 --cv_folds 5
```

### 🔹 2. รันการทดลองแยกตามโมเดลเดี่ยวๆ (Single Model Execution)
* **รันเฉพาะ XGBoost สำหรับ Task 1:**
  ```bash
  python run_classical_ml.py --task 1 --model XGBClassifier --use_gpu true --run_id run_02_autotune_hyperparams --n_trials 10 --cv_folds 5
  ```
* **รันเฉพาะ LightGBM สำหรับ Task 1:**
  ```bash
  python run_classical_ml.py --task 1 --model LGBMClassifier --use_gpu true --run_id run_02_autotune_hyperparams --n_trials 10 --cv_folds 5
  ```

### 🔹 3. รันการทดลองใน Task อื่นๆ (Tasks 2, 3, 4)
* **Task 2 (NER & Count Extraction):**
  ```bash
  python run_classical_ml.py --task 2 --use_gpu true --run_id run_02_autotune_hyperparams
  ```
* **Task 3 (Clinical Triage Classification - Pediatric & Adult):**
  ```bash
  python run_classical_ml.py --task 3 --use_gpu true --run_id run_02_autotune_hyperparams
  ```
* **Task 4 (End-to-End Pipeline Benchmark บน Luna Dataset):**
  ```bash
  python run_classical_ml.py --task 4 --use_gpu true --run_id run_03_final_luna_benchmark --latency_runs 1000
  ```
* **รันการทดลองทั้งหมดตั้งแต่ Task 1 ถึง Task 4:**
  ```bash
  python run_classical_ml.py --task all --model all --use_gpu true --run_id run_02_autotune_hyperparams --n_trials 10 --cv_folds 5
  ```

---

## ⚙️ รายละเอียดพารามิเตอร์ CLI (`argparse`)

| พารามิเตอร์ | ค่าเริ่มต้น | รายละเอียด |
| :--- | :---: | :--- |
| `--task` | `all` | เลือก Task ที่ต้องการรัน (`1`, `2`, `3`, `3.1`, `3.2`, `3.3`, `4`, หรือ `all`) |
| `--model` | `all` | เลือกชื่อโมเดล (เช่น `XGBClassifier`, `LGBMClassifier`, `LogisticRegression`, หรือ `all`) |
| `--use_gpu` | `true` | เปิด/ปิดการเร่งความเร็วด้วย GPU CUDA (`true`/`false`) |
| `--run_id` | `run_02_autotune_hyperparams` | ชื่อไดเรกทอรีเก็บผลลัพธ์การทดลอง |
| `--n_trials` | `15` | จำนวนรอบ Optuna Auto-Tuning สุ่มจูนพารามิเตอร์ต่อโมเดล |
| `--cv_folds` | `5` | จำนวน Folds สำหรับ Cross-Validation |
| `--latency_runs` | `500` | จำนวนรอบวัดความเร็วเวลาประมวลผล Latency Protocol |
| `--force` | `false` | บังคับรันใหม่ทับไฟล์ Checkpoint เดิม |

---

## 📁 โครงสร้างโฟลเดอร์ผลลัพธ์ (Results Directory Structure)

```
results/classical_ml/tuning_history/run_02_autotune_hyperparams/
├── best_configs/            # พารามิเตอร์ที่ดีที่สุดแยกรายโมเดล (JSON)
├── graphs/                  # กราฟสรุปผลวิจัย 6 รูปแบบ (PNG)
├── logs/                    # Optuna trial raw logs (JSON)
├── stat_tests/              # ผลทดสอบทางสถิติ McNemar + Holm & Wilcoxon (JSON)
├── task1_summary.csv        # ตารางสรุปคะแนน F1, F2, MCC, Latency ของ Task 1
├── task2_summary.csv        # ตารางสรุปคะแนน Extraction ของ Task 2
├── task3_summary.csv        # ตารางสรุปคะแนน Clinical Triage ของ Task 3
└── task4_e2e_results/       # ผลลัพธ์ประเมินพายป์ไลน์รวม Task 4
```
