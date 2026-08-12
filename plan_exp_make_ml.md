# แผนการทดลองรายละเอียดสูง: ระบบประมวลผลภัยพิบัติด้วย Classical Machine Learning & Deep Learning (Hybrid GPU/CPU Architecture)

## 1. บทนำและเป้าหมายของงานวิจัย (Overview & Research Objectives)

งานวิจัยนี้มีเป้าหมายเพื่อสร้างและประเมินระบบประมวลผลข้อความภัยพิบัติภาษาไทยจากโซเชียลมีเดีย โดยหลีกเลี่ยงการใช้ Large Language Models (LLMs) ขนาดใหญ่แบบ API แต่ใช้สถาปัตยกรรม **Classical Machine Learning, Rule-Based Engine และ BiLSTM-CRF** 

* **ช่วงการฝึกสอน (Training Phase):** ใช้ **Hybrid GPU/CPU Training Acceleration** เร่งความเร็วโมเดลที่รองรับ CUDA (เช่น `XGBoost`, `LightGBM`, `CatBoost`, `cuML`, `PyTorch BiLSTM-CRF`) ร่วมกับการประมวลผลแบบ Multi-threading บน CPU สำหรับโมเดล Scikit-Learn และ CRF
* **ช่วงการวัดผลเวลาประมวลผล (Inference Latency Evaluation):** ทำการวัดผลและเปรียบเทียบเวลาประมวลผลจริง (Empirical Inference Latency) อย่างรัดกุมผ่านระเบียบวิธีวิศวกรรม ทั้งในสภาพแวดล้อม **GPU** (สำหรับศูนย์ประมวลผลกลาง) และ **CPU** (สำหรับสภาพแวดล้อมสเปกทั่วไป/เครื่องฝั่งภาคสนาม) เพื่อศึกษาความแตกต่างของความเร็วและ Throughput ของแต่ละสถาปัตยกรรม

---

## 2. ยุทธศาสตร์ข้อมูล 5-Fold Cross-Validation การป้องกัน Data Leakage และการประเมิน Cross-Generator Performance Gap

### 2.1 ข้อกำหนดป้องกัน Data Leakage อย่างเข้มงวด (Strict Data Leakage Prevention Directives)
เพื่อป้องกันปัญหาข้อมูลรั่วไหล (Data Leakage) ระหว่างข้อมูลส่วนการฝึกและส่วนประเมินผล การทำระบบ Pipeline ใน Implementation จะถูกล็อกด้วยกฎเหล็ก 2 ข้อ:

1. **TF-IDF Fitting ภายใน CV Fold เท่านั้น:** 
   * ห้ามทำการ `vectorizer.fit(all_data)` บนชุดข้อมูลทั้งหมดก่อนแบ่ง Fold เด็ดขาด เพราะจะทำให้คำศัพท์ (Vocabulary) และค่า IDF จาก Validation Fold รั่วไหลเข้าไปยัง Training Fold
   * **โครงสร้างที่ถูกต้อง:** ใช้ `sklearn.pipeline.Pipeline` หรือลูป Fold ที่ทำการ `TF-IDF.fit_transform()` เฉพาะข้อมูล `X_train` ใน Fold นั้นๆ แล้วจึง `TF-IDF.transform()` ข้อมูล `X_val` และ `X_test` เท่านั้น
2. **ลำดับการทำงานของ Optuna Auto-Tuning (Optuna Outside CV Loop):**
   * Optuna จะสุ่มเลือกค่า Hyperparameters $\rightarrow$ ส่งเข้าลูป 5-Fold CV $\rightarrow$ ในแต่ละ Fold จะทำ `TF-IDF.fit()` บน `X_train` และประเมินบน `X_val` $\rightarrow$ บันทึกและส่งคะแนนทั้ง **$\text{Mean } F_1$** และ **$\text{Mean } F_2$** กลับไปให้ Optuna

```
                               ┌──────────────────────────────────────────┐
                               │ Optuna Trial (Suggest Hyperparameters)   │
                               └────────────────────┬─────────────────────┘
                                                    │
                                                    ▼
                               ┌──────────────────────────────────────────┐
                               │ 5-Fold Cross-Validation Loop             │
                               │ ┌──────────────────────────────────────┐ │
                               │ │ Fold 1..5:                           │ │
                               │ │ ├── X_train ──► TF-IDF.fit_transform │ │
                               │ │ └── X_val   ──► TF-IDF.transform     │ │
                               │ └──────────────────────────────────────┘ │
                               └────────────────────┬─────────────────────┘
                                                    │
                                                    ▼
                               ┌──────────────────────────────────────────┐
                               │ Calculate Mean 5-Fold F1 & F2 Scores     │
                               └────────────────────┬─────────────────────┘
                                                    │
                                                    ▼
                               ┌──────────────────────────────────────────┐
                               │ Optuna Sampler Updates Best Parameters   │
                               └──────────────────────────────────────────┘
```

### 2.2 กำหนดการทำ Cross-Validation แยกตามคลาสของ Task
1. **Task 1 (Classification), Task 3.2 (Pediatric Triage) & Task 3.3 (Adult Triage):**
   * ใช้ **5-Fold Stratified K-Fold (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`)** เพื่อรักษาสัดส่วนคลาสหมวดหมู่ (`help_request`, `RED`/`YELLOW`/`GREEN`) ให้เท่ากันในทุก Fold
2. **Task 2 (Quantity / Count Predictions: `gt_dead`, `gt_critical`, `gt_food`, ฯลฯ):**
   * **แนวทาง A (Binned Categorical Classification):** แปลงตัวเลขนับเป็น Class ช่วงแบบจำกัด (`0`, `1`, `2`, `3+`) $\rightarrow$ ใช้ **5-Fold Stratified K-Fold**
   * **แนวทาง B (Continuous Numerical Regression):** ใช้ตัวเลขนับต่อเนื่องจริง ($0, 1, 2, 7, 13\dots$) $\rightarrow$ ใช้ **5-Fold Standard K-Fold** ร่วมกับ Regression Models (`XGBRegressor`, `LGBMRegressor`, `CatBoostRegressor`, `Ridge`) และประเมินผลด้วย `MAE` / `RMSE`
3. **Final Benchmark Test Dataset (Luna Dataset - Held-out Test):** `dataset/gpt_5_6_luna_paired_synthetic_ner_dataset.csv`
   * ทดสอบประเมินประสิทธิภาพขั้นสุดท้ายเพื่อวัด **Cross-Generator Performance Gap** ทั้งแบบแยก Task (Task 1, 2, 3.1, 3.2, 3.3) และแบบพายป์ไลน์รวมครบวงจร (**Task 4: End-to-End Pipeline**)

### 2.3 ขั้นตอนการทดลอง 4 ระยะ (4-Stage Execution Pipeline)

* **Stage 1: Pipeline-based 5-Fold CV & Auto-Tuning**
  * สุ่มค้นหา Hyperparameters สำหรับทุกโมเดลด้วย **Optuna** โดยใช้ `Pipeline(TFIDF, Model)` บน 5-Fold CV ตามประเภท Task (Task 1, 2, 3.1, 3.2, 3.3)
  * บันทึกคะแนนเฉลี่ยทั้ง **$\text{Mean } F_1 \pm \text{Std}$** และ **$\text{Mean } F_2 \pm \text{Std}$** ร่วมกันทุกรอบ
  * **บันทึกประวัติการจูน (Tuning History Logs):** เก็บไฟล์ JSON/CSV บันทึกพารามิเตอร์ทุกชุดที่ลอง และค่าพารามิเตอร์ที่ดีที่สุด (`best_params_`)
* **Stage 2: Model Re-fitting on Full Gemini Dataset**
  * นำค่า Hyperparameters ที่ดีที่สุด (`best_params_`) ของแต่ละโมเดล มาทำ `Pipeline(TFIDF.fit(), Model.fit())` บนข้อมูลครบ 100% ของ Gemini Dataset
* **Stage 3: End-to-End Pipeline Assembly & Final Benchmark Test on Luna Dataset**
  * นำโมเดลที่ดีที่สุดจาก Task 1, Task 2, Task 3.1, Task 3.2 (Pediatric Triage: เด็ก $\le 12$ ปี) และ Task 3.3 (Adult Triage: ผู้ใหญ่ $> 12$ ปี) มาประกอบรวมกันเป็น **Task 4: End-to-End Pipeline System**
  * นำพายป์ไลน์รวมไปประเมินบน `gpt_5_6_luna_paired_synthetic_ner_dataset.csv` (100% Unseen Data) โดยเปรียบเทียบกับ Ground Truth ครบทุก Field และเปรียบเทียบกับ LLM Full Agent Baseline
* **Stage 4: Rigorous Statistical Significance Testing (McNemar + Holm Correction)**
  * ทดสอบนัยสำคัญทางสถิติของการจำแนกบนชุด Luna ด้วย **McNemar's Test** สำหรับคู่โมเดล Top-3 ($A$ vs $B$, $A$ vs $C$, $B$ vs $C$) 
  * ปรับแก้ค่า $p$-value ด้วย **Holm-Bonferroni Correction** เพื่อควบคุม Family-wise Error Rate (FWER)
  * ใช้ **Wilcoxon Signed-Rank Test** เฉพาะสำหรับการเปรียบเทียบค่าความผิดพลาดต่อเนื่องรายตัวอย่างทดสอบ (**Sample-wise Continuous Residuals / Absolute Errors MAE**) ข้ามโมเดลใน Task 2 Regression (ไม่ใช้ Wilcoxon กับ Latency)
  * วิเคราะห์ส่วนต่างประสิทธิภาพเนื่องจาก Domain Shift ข้าม Generator (**Cross-Generator Performance Gap**: $\Delta F1 = F1_{\text{Gemini\_CV}} - F1_{\text{Luna\_Test}}$ และ $\Delta F2 = F2_{\text{Gemini\_CV}} - F2_{\text{Luna\_Test}}$)

---

## 3. การจัดเตรียม Feature Representation, Baseline Models และ Hardware Mapping

### 3.1 Feature Representation Strategies (การแปลงข้อความภาษาไทย)
1. **TF-IDF Word-Level:** ตัดคำภาษาไทยด้วย `PyThaiNLP` (engine="newmm") สกัด Unigram + Bigram (1-2 words)
2. **TF-IDF Char-Level:** สกัด N-grams ระดับตัวอักษร (2-4 chars) เพื่อจัดการปัญหารากคำ การสะกดผิด และคำที่ไม่ได้อยู่ในพจนานุกรม
3. **Hybrid Feature Union:** รวมคุณลักษณะจากทั้ง Word-Level และ Char-Level เข้าด้วยกัน (กำหนด Max Features 5,000–10,000)

> 🔒 **การันตีปราศจาก Data Leakage:** การประกอบ Feature Union ทั้งหมดจะถูกห่อหุ้มใน `sklearn.pipeline.Pipeline` หรือ `imblearn.pipeline.Pipeline` เพื่อให้ `fit()` เกิดขึ้นเฉพาะข้อมูลฝั่ง Train ในแต่ละ Fold เท่านั้น

### 3.2 โมเดลเกณฑ์มาตรฐาน (Baselines)
เพื่อพิสูจน์ความคุ้มค่าของการใช้โมเดล ML การทดลองจะเปรียบเทียบกับ 2 Baselines พื้นฐาน:
1. **Baseline 0a (Dummy Majority Classifier / Mean Regressor):** ทำนายคลาสที่พบมากที่สุดหรือค่าเฉลี่ยตัวเลขนับ (`DummyClassifier` / `DummyRegressor`)
2. **Baseline 0b (Simple Keyword Rule-Based):** ใช้ Regex จับคำสัญญาณภัยพิบัติและตัวเลขนับพื้นฐาน (เช่น *"ช่วยด้วย"*, *"ขอเรือ"*, *"เสียชีวิต (\d+)"*)

### 3.3 รายชื่อโมเดล และ ตารางตรวจสอบฮาร์ดแวร์จริง (Hardware & Library Mapping Table)

เพื่อความถูกต้องทางวิชาการและโปร่งใสสูงสุด ตารางด้านล่างแสดงการแมป Library และสถานะการรองรับ GPU จริงของแต่ละโมเดล:

| ชื่อโมเดล (Model Name) | Library ที่ใช้จริง | สถานะการรองรับ GPU | หมายเหตุทางเทคนิค |
| :--- | :--- | :---: | :--- |
| **Baselines** | | | |
| 0a. `DummyClassifier` / `DummyRegressor` | `scikit-learn` | ❌ CPU Only | Baseline ทำนายคลาสส่วนมาก / ค่าเฉลี่ย |
| 0b. `Simple Keyword Rules` | Python Native / Regex | ❌ CPU Only | Baseline กฎคีย์เวิร์ดพื้นฐาน |
| **Linear & SVM Models** | | | |
| 1. `LogisticRegression` / `Ridge` | `cuml` / `scikit-learn` | ✅ GPU (cuML) | สลับเป็น CPU (scikit-learn) ได้อัตโนมัติ |
| 2. `LinearSVC` / `LinearSVR` | `cuml` / `scikit-learn` | ✅ GPU (cuML) | cuML รองรับ LinearSVC/SVR |
| 3. `SVC` / `SVR (kernel='linear')` | `cuml` / `scikit-learn` | ✅ GPU (cuML) | cuML `cuml.svm` |
| 4. `SVC` / `SVR (kernel='rbf')` | `cuml` / `scikit-learn` | ✅ GPU (cuML) | cuML `cuml.svm` |
| 5. `SVC` / `SVR (kernel='poly')` | `cuml` / `scikit-learn` | ✅ GPU (cuML) | cuML `cuml.svm` |
| 6. `SVC` / `SVR (kernel='sigmoid')` | `cuml` / `scikit-learn` | ✅ GPU (cuML) | cuML `cuml.svm` |
| 7. `RidgeClassifier` / `Ridge` | `cuml` / `scikit-learn` | ⚠️ GPU (cuML Threshold) | cuML `Ridge` สำหรับ Regression / Classification |
| 8. `PassiveAggressiveClassifier/Regressor` | `scikit-learn` | ❌ CPU Multi-thread | sklearn ไม่มี cuML implementation (รัน CPU) |
| 9. `SGDClassifier` / `SGDRegressor` | `cuml` / `scikit-learn` | ✅ GPU (cuML) | cuML `MBSGDClassifier/Regressor` |
| **Naive Bayes & Distance** | | | |
| 10. `MultinomialNB` | `cuml` / `scikit-learn` | ✅ GPU (cuML) | cuML `cuml.naive_bayes.MultinomialNB` |
| 11. `ComplementNB` | `cuml` / `scikit-learn` | ✅ GPU (cuML) | cuML `cuml.naive_bayes.ComplementNB` |
| 12. `KNeighborsClassifier` / `Regressor` | `cuml` / `scikit-learn` | ✅ GPU (cuML) | cuML `cuml.neighbors` |
| 13. `MLPClassifier` / `MLPRegressor` | `PyTorch` / `scikit-learn` | ✅ GPU (PyTorch) | ใช้ PyTorch Neural Net บน CUDA |
| **Tree & Boosting** | | | |
| 14. `DecisionTreeClassifier` / `Regressor` | `cuml` / `scikit-learn` | ❌ CPU Multi-thread | sklearn ใช้ CPU (`n_jobs=-1`) |
| 15. `RandomForestClassifier` / `Regressor` | `cuml` / `scikit-learn` | ✅ GPU (cuML) | cuML `RandomForestClassifier/Regressor` |
| 16. `ExtraTreesClassifier` / `Regressor` | `scikit-learn` | ❌ CPU Multi-thread | sklearn ใช้ CPU (`n_jobs=-1`) |
| 17. `AdaBoostClassifier` / `Regressor` | `scikit-learn` | ❌ CPU Multi-thread | sklearn ไม่มี GPU support (รัน CPU) |
| 18. `GradientBoostingClassifier` / `Regressor` | `scikit-learn` | ❌ CPU Multi-thread | sklearn ไม่มี GPU support (รัน CPU) |
| 19. `XGBClassifier` / `XGBRegressor` | `xgboost` | 🚀 GPU Native | `tree_method='hist'`, `device='cuda'` |
| 20. `LGBMClassifier` / `LGBMRegressor` | `lightgbm` | 🚀 GPU Native | `device='gpu'` |
| 21. `CatBoostClassifier` / `CatBoostRegressor` | `catboost` | 🚀 GPU Native | `task_type='GPU'` |
| **Sequence Models & NER** | | | |
| 22. `sklearn-crfsuite (CRF)` | `sklearn-crfsuite` | ❌ CPU Only | CRF implementation เป็น C++ CPU-only ล้วน |
| 23. `BiLSTM-CRF` | `PyTorch` | 🚀 GPU Native | `torch.device('cuda')` |

---

## 4. รายละเอียดการประเมินผลและระเบียบวิธีการวัด Latency (Metrics & Measurement Protocol)

### 4.1 ชุดตัววัดประสิทธิภาพตามมิติต่างๆ (Categorized Metrics Suite)

#### 🅰️ Classification & Disaster Relevance Metrics (สำหรับ Task 1, Task 3.2 และ Task 3.3)
* **Accuracy, Precision, Recall**
* **F1-Score (Micro / Macro / Weighted):** ค่าเฉลี่ยฮาร์โมนิกระหว่าง Precision และ Recall (วัดสมดุลทั่วไป)
* **F2-Score ($F_\beta$ โดย $\beta=2$):** **ตัววัดเน้นความปลอดภัยชีวิต!** ให้ค่าน้ำหนัก Recall สูงเป็น 2 เท่าของ Precision ($F_2 = \frac{5 \cdot P \cdot R}{4 \cdot P + R}$) เนื่องจาก False Negative (หลุดคนป่วย) มีอันตรายร้ายแรงกว่า False Positive
* **MCC (Matthews Correlation Coefficient)** & **Cohen's Kappa ($\kappa$)**
* **PR-AUC, ROC-AUC, Log Loss**

#### 🅱️ NER & Entity/Count Extraction Metrics (สำหรับ Task 2)
* **Exact Match (EM) Rate**, **Partial Match Jaccard**, **Normalized Levenshtein Distance**
* **MAE (Mean Absolute Error)** & **RMSE** สำหรับ Count Regression

#### 🅲️ Clinical Triage & Patient Safety Metrics (สำหรับ Task 3.2 และ Task 3.3)
* **Alignment-based Patient Match Rate** & **Quadratic Weighted Kappa (QWK)**
* **Critical Under-Triage Rate (UR) พร้อม 95% CI** (คำนวณช่วงความเชื่อมั่น 95% ด้วย Bootstrapping) & **Over-Triage Rate (OR)**

---

### 4.2 ระเบียบวิธีการวัดเวลาประมวลผลอย่างเคร่งครัด (Inference Latency Measurement Protocol)

เพื่อให้ค่าเวลาประมวลผล (Inference Latency) มีความเที่ยงตรง สอดคล้องตามมาตรฐานวิศวกรรม และสามารถเปรียบเทียบข้ามโมเดลได้อย่างชอบธรรม การวัดผลจะประเมินผ่าน **Empirical Performance Distribution (ไม่ใช้ Hypothesis Testing สถิติเชิงเปรียบเทียบกับ Latency)** ตาม **Protocol 5 ข้อ** ต่อไปนี้:

```
[1. Warm-up Phase] ──► [2. Single-Sample Input] ──► [3. End-to-End Pipeline] ──► [4. CUDA Sync] ──► [5. Empirical Distribution]
(100 Iterations)       (Batch Size = 1)             (Text -> Vector -> Output)     (torch.cuda.sync)   (Mean, p50, p95, p99, QPS)
```

1. **Hardware & Single-Thread Enforcement (การคุมสภาพแวดล้อม):**
   * บันทึกรุ่น CPU, RAM, GPU และเวอร์ชันของ CUDA/PyTorch อย่างละเอียด
   * สำหรับการวัดผลบน CPU จะต้องกำหนดให้ใช้ 1 Core (`torch.set_num_threads(1)` และ `n_jobs=1`) เพื่อจำลองประสิทธิภาพของระบบเดี่ยว (Single-thread Baseline)
2. **Warm-up Phase (การวอร์มอัพระบบก่อนวัดจริง):**
   * ทำการรัน Inference เปล่าจำนวน **100 Iterations** ก่อนเริ่มบันทึกเวลาจริง เพื่อให้ GPU CUDA Context สมบูรณ์, PyTorch JIT/CUDNN Benchmarks เข้าสู่สภาวะคงที่ และ CPU Cache พร้อมทำงาน
3. **Real-Time Single-Sample Batching (Batch Size = 1):**
   * กำหนด **Batch Size = 1** สำหรับทุกการทดสอบ เพื่อจำลองสถานการณ์จริงที่มีข้อความขอความช่วยเหลือเข้าสู่ระบบแบบสตรีมมิ่งทีละข้อความ (Real-time Disaster Streaming)
4. **End-to-End Pipeline Scope (ขอบเขตการวัดแบบครอบคลุมทั้งกระบวนการ):**
   * เวลาที่วัด (`perf_counter`) จะต้องครอบคลุมตั้งแต่ **ข้อความดิบ (Raw Text) $\rightarrow$ การตัดคำ PyThaiNLP $\rightarrow$ การแปลง TF-IDF Vector $\rightarrow$ การประมวลผลโมเดล $\rightarrow$ การส่งคำตอบ Output**
   * สำหรับการวัดบน GPU จะต้องเรียก `torch.cuda.synchronize()` ทั้งก่อนเริ่มจับเวลาและหลังประมวลผลเสร็จ เพื่อป้องกันปัญหาการวัดเวลาคลาดเคลื่อนจาก Asynchronous CUDA Execution
5. **Empirical Distribution & Percentile Reporting ($N = 1,000$ Runs):**
   * เนื่องจากเวลาประมวลผลจากการรันบนเครื่องเดียวกันมีความไม่เป็นอิสระต่อกัน (Time-series Dependent / Hardware Cache / OS CPU Scheduling) จึง**ไม่ใช้ Hypothesis Testing (เช่น Wilcoxon / t-test) กับ Latency** 
   * ให้รายงานผลเป็น **การกระจายเชิงประจักษ์ (Empirical Distribution)** ผ่านค่าสถิติวัดตำแหน่ง:
     * **Mean Latency $\pm$ Std** (มิลลิวินาที)
     * **p50 (Median Latency):** ค่ามัธยฐานเวลาประมวลผล
     * **p95 (95th Percentile Latency):** เวลาประมวลผลช้าสุดในขอบเขต 95%
     * **p99 (99th Percentile Latency):** เวลาประมวลผลช้าสุดในเคสสุดขั้ว (Tail Latency)
     * **Throughput (QPS):** จำนวนข้อความที่ประมวลผลได้ต่อ 1 วินาที ($QPS = 1000 / \text{Mean Latency}$)

---

## 5. รายละเอียดการทดลองแบบย่อยแยกตาม Task (Detailed Task Methodology)

---

### 📌 TASK 1: Disaster Tweet Classification (จำแนกประเภทข้อความ)

#### 5.1 วัตถุประสงค์
เพื่อทดสอบประเมินประสิทธิภาพในการกรองข้อความว่า ข้อความโซเชียลมีเดียเป็นข้อความขอความช่วยเหลือฉุกเฉิน (`help_request`) หรือข้อความทั่วไป (`other`)

#### 5.2 การตั้งค่าการทดลอง (Experimental Setup)
* **Target Label:** `gt_is_help_request` (Binary: 0/1) หรือ `gt_classification_category` (`help_request` vs `other`)
* **Tuning Protocol:**
  * เทรนด้วย **5-Fold Stratified Cross-Validation (`random_state=42`)** ผ่าน `Pipeline(TFIDF, Model)` บน Gemini Dataset
  * บันทึกคะแนนประวัติ Optuna ทั้ง **F1-Score** และ **F2-Score** ร่วมกันในแต่ละ Trial
  * บันทึก Best Config ลงใน `results/classical_ml/tuning_history/run_02_autotune_hyperparams/best_configs/task1_best_params.json`
* **Final Benchmark Test:** นำ Pipeline ที่ Re-fit ด้วย Best Config บน 100% Gemini Dataset ไปทำนายบน Luna Dataset (`gpt_5_6_luna_paired_synthetic_ner_dataset.csv`)

#### 5.3 การวัดผล (Evaluation Metrics)
* Primary: **`Mean 5-Fold F1-Score ± Std`** และ **`Mean 5-Fold F2-Score ± Std`** (รายงานทั้ง 2 ตัววัดเคียงคู่กัน), `MCC`
* Secondary: `Accuracy`, `Precision`, `Recall`, `ROC-AUC`, `Log Loss`
* Statistical Significance Test: **McNemar's Test** ร่วมกับ **Holm-Bonferroni Correction** ($p_{\text{adj}} < 0.05$) ข้ามคู่โมเดล Top-3 บนชุด Luna Dataset
* Computational Protocol: `Empirical Latency Distribution (Mean, p50, p95, p99, QPS)` ทั้งบน GPU และ CPU

---

### 📌 TASK 2: NER & Entity/Count Extraction (ดึงพิกัด ชื่อ เบอร์โทร จำนวน)

#### 5.1 วัตถุประสงค์
เพื่อทดสอบเปรียบเทียบประสิทธิภาพในการสกัดข้อมูลเฉพาะ (Named Entities) และการทำนายจำนวนนับ (Quantity Extraction) จากข้อความขอความช่วยเหลือ

#### 5.2 การตั้งค่าการทดลอง (Model vs Rule-Based vs Hybrid & Classification vs Regression)
การทำทำนายจำนวนนับ (`gt_dead`, `gt_critical`, `gt_urgent`, `gt_safe`, `gt_child`, `gt_bedridden`, `gt_item_firstaid`, `gt_item_food`, `gt_item_energy`) จะถูกแยกเป็น 2 รูปแบบการทดลองให้ชัดเจน:

* **Approach A: Rule-Based Engine (Regex + Gazetteers)**
  * **Phone Number:** Regex `0[689]\d{8}` และ `0\d{1,2}-\d{3}-\d{4}`
  * **Coordinates & Maps:** Regex ทศนิยม Lat/Lng `1\d\.\d{4,},\s*100\.\d{4,}` และ Google Maps URL
  * **Location Name:** PyThaiNLP Gazetteer (รายชื่อจังหวัด อำเภอ ตำบล) + Custom Landmark Gazetteers
  * **Victim & Item Counts:** Keyword Pattern Matching (เช่น `"เสียชีวิต (\d+)"`, `"ผู้ป่วย (\d+)"`, `"อาหาร (\d+)"`)
* **Approach B1: Binned Categorical Classification Models (17 Models Benchmark)**
  * แปลงตัวเลขนับเป็น Class ช่วงแบบจำกัด (`0`, `1`, `2`, `3+`) $\rightarrow$ ใช้ **5-Fold Stratified K-Fold** บน `Pipeline(TFIDF, Classifier)` และรายงานผลทั้ง **F1-Score** และ **F2-Score**
* **Approach B2: Continuous Numerical Regression Models (17 Regressor Models Benchmark)**
  * ใช้ตัวเลขนับต่อเนื่องจริง ($0, 1, 2, 7, 13\dots$) $\rightarrow$ ใช้ **5-Fold Standard K-Fold** บน `Pipeline(TFIDF, Regressor)` (`XGBRegressor`, `LGBMRegressor`, `CatBoostRegressor`, `Ridge`, `RandomForestRegressor`) และประเมินด้วย **`MAE`** / **`RMSE`**
* **Approach B3: Token Classification (CRF & Sliding Window Classifiers)**
  * สกัดป้ายกำกับคำ (`B-LOC`, `B-PER`, `B-PHONE`) ด้วย `sklearn-crfsuite` และ 17 Classifiers
* **Approach C: Hybrid System (Best ML Model + Rules Engine) ⭐**
  * ใช้ **ML Model/CRF** ที่ผ่านการจูนแล้ว สกัด *ชื่อสถานที่* และ *ชื่อบุคคล*
  * ใช้ **Rule-Based/Regex/Best Regressor** สกัด *เบอร์โทรศัพท์*, *พิกัด Lat/Lng* และ *จำนวนนับต่างๆ*
* **Final Benchmark Test:** รันการประเมินเทียบทุก Approaches บน Luna Dataset (`gpt_5_6_luna_paired_synthetic_ner_dataset.csv`)

#### 5.3 การวัดผล (Evaluation Metrics)
* `Exact Match (EM) Score` แยกราย Field: `location_em`, `map_url_em`, `lat_em`, `lng_em`, `victim_name_em`, `victim_phone_em`
* Text Similarity: `Partial Match Jaccard`, `Normalized Levenshtein Distance`
* Quantity Error (Regression B2): `MAE`, `RMSE` (ทดสอบความแตกต่างรายข้อความด้วย **Wilcoxon Signed-Rank Test** ข้ามโมเดล)
* Operational: `Overall Average EM`, `Empirical Latency Distribution (Mean, p50, p95, p99, QPS)`

---

### 📌 TASK 3: People Extraction & Clinical Triage Classification (แยกผู้ป่วย & คัดกรองสี 2 โมเดล)

#### 5.1 วัตถุประสงค์
เพื่อทดสอบเปรียบเทียบการสกัดผู้ป่วยแต่ละราย (Individual Victims) และการประเมินระดับความรุนแรงทางการแพทย์ตามมาตรฐาน IITT (Interagency Integrated Triage Tool) โดยแยกโมเดลคัดกรองทางการแพทย์เป็น **2 โมเดลเฉพาะทาง (Pediatric vs Adult Triage)** สอดคล้องกับโครงสร้างของ Full Agent (`agent3_2_pediatric_triage.py` และ `agent3_3_adult_triage.py`)

#### 5.2 การตั้งค่าการทดลองย่อย (Sub-task Experiments)

##### 🔹 Sub-task 3.1: People Extraction (แยกผู้ป่วย & สกัดอาการ)
* **Method 3.1a (Rule-based Clause Splitter):** PyThaiNLP ตัดอนุประโยคตามคำเชื่อม (`และ`, `กับ`, `ส่วน`, `\n`) + Dictionary Matcher
* **Method 3.1b (BiLSTM-CRF Extractor - PyTorch GPU):** เทรน **BiLSTM-CRF** บน GPU CUDA (5-Fold Stratified CV บน Gemini Dataset) สกัด `B-PER`, `B-AGE`, `B-SYMPTOM`

##### 🔹 Sub-task 3.2: Pediatric Clinical Triage Classification (โมเดลคัดกรองกุมารเวชกรรม: เด็ก $\le 12$ ปี) ⭐
* **Target Group:** ผู้ป่วยที่เป็นเด็ก (`gt_child > 0` หรือมีอายุระบุ $\le 12$ ปี ตามเกณฑ์กุมารเวชกรรม)
* **Target Class:** RED / YELLOW / GREEN (วัดผลตามเกณฑ์ Pediatric IITT)
* **Methods Tested:**
  * **Method 3.2a (Pediatric IITT Rule Engine):** จับแมตช์อาการป่วยของเด็กกับเกณฑ์ Pediatric IITT
  * **Method 3.2b (17 Classical ML Classifiers Benchmark):** เทรน 17 ML Models บนข้อความอาการเด็กด้วย 5-Fold Stratified CV
  * **Method 3.2c (Hybrid BiLSTM Embedding + GBDT Classifiers):** สกัด Vector จาก BiLSTM ป้อนให้ XGBoost/LGBM/CatBoost บน GPU

##### 🔹 Sub-task 3.3: Adult Clinical Triage Classification (โมเดลคัดกรองผู้ใหญ่/ทั่วไป: อายุ $> 12$ ปี) ⭐
* **Target Group:** ผู้ป่วยผู้ใหญ่หรือผู้ป่วยทั่วไปที่มีอายุ $> 12$ ปี
* **Target Class:** RED / YELLOW / GREEN (วัดผลตามเกณฑ์ Adult IITT)
* **Methods Tested:**
  * **Method 3.3a (Adult IITT Rule Engine):** จับแมตช์อาการป่วยผู้ใหญ่กับเกณฑ์ Adult IITT
  * **Method 3.3b (17 Classical ML Classifiers Benchmark):** เทรน 17 ML Models บนข้อความอาการผู้ใหญ่ด้วย 5-Fold Stratified CV
  * **Method 3.3c (Hybrid BiLSTM Embedding + GBDT Classifiers):** สกัด Vector จาก BiLSTM ป้อนให้ XGBoost/LGBM/CatBoost บน GPU

#### 5.3 การวัดผล (Evaluation Metrics สำหรับ 3.2 และ 3.3)
* Primary: **`Weighted F1-Score`**, **`Weighted F2-Score`**, `Quadratic Weighted Kappa (QWK)`, `Critical Under-Triage Rate (UR) พร้อม 95% CI` (รายงานแยกรายโมเดล 3.2 และ 3.3)
* Secondary: `Triage Accuracy`, `Over-Triage Rate (OR)`, `Alignment Patient Match Rate`
* Statistical Significance Test: **McNemar's Test** ร่วมกับ **Holm-Bonferroni Correction** ($p_{\text{adj}} < 0.05$) ข้ามโมเดล Top-3 บนชุด Luna Dataset
* Operational: `Empirical Latency Distribution (GPU & CPU)`

---

### 📌 TASK 4: End-to-End Integrated Classical ML Pipeline Benchmark (ประเมินพายป์ไลน์รวมทั้งระบบ) ⭐

#### 5.1 วัตถุประสงค์
เพื่อนำโมเดลและส่วนประกอบที่ดีที่สุดจากการทดลองใน Task 1, Task 2, Task 3.1, Task 3.2 (Pediatric Triage: เด็ก $\le 12$ ปี) และ Task 3.3 (Adult Triage: ผู้ใหญ่ $> 12$ ปี) มาประกอบรวมกันเป็น **Full Integrated Disaster NLP Pipeline** ทำงานร้อยต่อกันอย่างสมบูรณ์แบบ (End-to-End Streaming Flow) สำหรับทดสอบเปรียบเทียบประสิทธิภาพและความเร็วในสถานการณ์จริงเทียบกับระบบ **LLM Full Agent (`run_evaluation.py`)**

#### 5.2 สถาปัตยกรรมสายการประมวลผล E2E Flow (E2E Pipeline Architecture)

```
                       ┌───────────────────────────────────────────────────────────┐
                       │ Raw Social Media Tweet Input                              │
                       └─────────────────────────────┬─────────────────────────────┘
                                                     │
                                                     ▼
                       ┌───────────────────────────────────────────────────────────┐
                       │ Step 1: Task 1 Best Classifier (e.g. XGBoost / LinearSVC) │
                       │ Is Help Request?                                          │
                       └──────────────┬─────────────────────────────┬──────────────┘
                                      │ Yes                         │ No
                                      ▼                             ▼
┌───────────────────────────────────────────────────────────┐ ┌────────────────────┐
│ Step 2: Task 2 Best NER & Count Models + Rules Engine     │ │ Classification:    │
│ Extract Location, Map URL, Lat/Lng, Phone, Victim/Item    │ │ "other"            │
└─────────────────────────────┬─────────────────────────────┘ └────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│ Step 3.1: Task 3.1 Best People Extractor (BiLSTM-CRF)     │
│ Extract Victim List: [{name, age, symptoms_literal}]      │
└─────────────────────────────┬─────────────────────────────┘
                              │
              ┌───────────────┴──────────────────────────────┐
              │ Route by Patient Age (<=12 vs >12 years)     │
              ▼                                              ▼
┌──────────────────────────────────────────┐   ┌──────────────────────────────────────────┐
│ Step 3.2: Best Pediatric Triage Model    │   │ Step 3.3: Best Adult Triage Model        │
│ (Pediatric IITT: Child Age <= 12 Years)  │   │ (Adult IITT: Age > 12 Years / General)   │
│ (RED / YELLOW / GREEN)                   │   │ (RED / YELLOW / GREEN)                   │
└─────────────────────┬────────────────────┘   └─────────────────────┬────────────────────┘
                      │                                              │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
┌───────────────────────────────────────────────────────────┐
│ Final Output JSON (เทียบเคียงกับ Output Schema ของ Full Agent)│
└───────────────────────────────────────────────────────────┘
```

#### 5.3 การวัดผลและการเปรียบเทียบกับ LLM Full Agent (Evaluation Metrics & Comparison)
* **Overall E2E Field-level Alignment F1-Score & Accuracy:** สัดส่วนความถูกต้องในการทำนาย JSON สรุปผลสุดท้ายตรงกับ Ground Truth ทุกฟิลด์ (`gt_classification_category`, `gt_location_name`, `gt_lat`, `gt_lng`, `gt_victim_phone`, `gt_victims_json`)
* **Overall E2E Clinical Triage Accuracy & Under-triage Rate (UR with 95% CI):** อัตราความถูกต้องและอัตราการหลุดผู้ป่วยวิกฤตของทั้งพายป์ไลน์รวม
* **Cumulative E2E Inference Latency (p50, p95, p99):** เวลาประมวลผลรวมครบทุก Step ของ 1 ข้อความ ทั้งบน **GPU** และ **CPU**
* **Trade-off Comparison vs LLM Full Agent:** เปรียบเทียบความแตกต่างด้าน Accuracy/F1 vs Latency/QPS vs Cost (ค่าใช้จ่าย API) ระหว่าง **Classical ML E2E Pipeline** กับ **LLM Full Agent**

---

## 6. ตารางเปรียบเทียบผลลัพธ์ภาพรวมของทั้งระบบ (Final Benchmark Output Format Demo)

> ⚠️ **หมายเหตุ:** ตัวเลขในตารางด้านล่างเป็น **"ตัวอย่างรูปแบบตารางสำหรับแสดงผล (Mock Data Format Demonstration)"** สำหรับกำหนดโครงสร้างตารางที่จะใช้บันทึกผลการทดลองจริงหลังจากรันโปรแกรมเรียบร้อยแล้ว

| Task / Module | วิธีการที่เข้าร่วมทดลอง | Metric หลักที่ใช้วัด | ผลลัพธ์ Gemini 5-Fold CV (Mean ± Std) | Final Benchmark on Luna Test | Critical Under-Triage Rate (UR) (95% CI) | GPU Latency (p95) | CPU Latency (p95) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Task 1: Classification** | Baseline 0a: Dummy Majority | F1 / F2 / MCC | [To be measured] | [To be measured] | N/A | [To be measured] | [To be measured] |
| | Baseline 0b: Simple Rules | F1 / F2 / MCC | [To be measured] | [To be measured] | [To be measured] | N/A | [To be measured] |
| | Hybrid TF-IDF + Top ML (Tuned) | F1 / F2 / MCC | [To be measured] | [To be measured] | N/A | [To be measured] | [To be measured] |
| **Task 2: NER / Extraction** | Approach A (Rules) | Overall EM / Partial | [To be measured] | [To be measured] | N/A | N/A | [To be measured] |
| | Approach B1 (Binned Classifiers) | Overall EM / F1 / F2 | [To be measured] | [To be measured] | N/A | [To be measured] | [To be measured] |
| | Approach B2 (Numerical Regressors)| MAE / RMSE / EM | [To be measured] | [To be measured] | N/A | [To be measured] | [To be measured] |
| | Approach C (Hybrid ⭐) | Overall EM / Partial / MAE | [To be measured] | [To be measured] | N/A | [To be measured] | [To be measured] |
| **Task 3.1: People Extractor** | Rule Splitter | Patient Alignment F1 | [To be measured] | [To be measured] | N/A | N/A | [To be measured] |
| | BiLSTM-CRF | Patient Alignment F1 | [To be measured] | [To be measured] | N/A | [To be measured] | [To be measured] |
| **Task 3.2: Pediatric Triage**| Method 3.2a: Pediatric Rules | E2E F1 / F2 / QWK / UR | [To be measured] | [To be measured] | [To be measured] | N/A | [To be measured] |
| | Method 3.2c: BiLSTM + GBDT | E2E F1 / F2 / QWK / UR | [To be measured] | [To be measured] | [To be measured] | [To be measured] | [To be measured] |
| **Task 3.3: Adult Triage** | Method 3.3a: Adult Rules | E2E F1 / F2 / QWK / UR | [To be measured] | [To be measured] | [To be measured] | N/A | [To be measured] |
| | Method 3.3c: BiLSTM + GBDT | E2E F1 / F2 / QWK / UR | [To be measured] | [To be measured] | [To be measured] | [To be measured] | [To be measured] |
| **Task 4: E2E Pipeline ⭐** | Full Integrated Rules Pipeline | E2E F1 / UR / Latency | [To be measured] | [To be measured] | [To be measured] | N/A | [To be measured] |
| | **Best E2E Classical ML Pipeline**| **Overall E2E F1 / UR / QPS**| **[To be measured]** | **[To be measured]** | **[To be measured]** | **[To be measured]** | **[To be measured]** |
| | *Baseline Ref: LLM Full Agent* | *E2E F1 / UR / API Cost* | *[Ref Baseline]* | *[Ref Baseline]* | *[Ref Baseline]* | *~1500 ms (API)* | *N/A (Cloud)* |

---

## 7. โครงสร้างโฟลเดอร์บันทึกประวัติ การวิเคราะห์กราฟ และการทดสอบทางสถิติ

### 7.1 โครงสร้างโฟลเดอร์ (Directory Hierarchy)
```
results/classical_ml/
├── tuning_history/
│   ├── run_01_baseline_default/         # รอบที่ 1: รันพารามิเตอร์ Default เพื่อตั้งเป็นเกณฑ์วัดมาตรฐาน (Baseline)
│   │   ├── logs/                        # Raw CSV/JSON logs (task1_default.json, task2_default.json)
│   │   ├── graphs/                      # กราฟเปรียบเทียบของรอบที่ 1
│   │   └── summary_metrics.csv
│   │
│   ├── run_02_autotune_hyperparams/     # รอบที่ 2: รัน GPU Auto-Tuning (Optuna 5-Fold CV Pipeline) บน Gemini Dataset
│   │   ├── logs/                        # Optuna trial logs (trials_task1.json, trials_task2.json, ...)
│   │   ├── best_configs/                # best_params_task1.json, best_params_task2.json, ...
│   │   ├── graphs/                      # กราฟวิเคราะห์การจูน (Optimization History, Feature Importance)
│   │   └── summary_metrics.csv
│   │
│   └── run_03_final_luna_benchmark/     # รอบที่ 3: Re-fit โมเดลตัวสมบูรณ์ แล้วทดสอบบน Luna Dataset (Out-of-Distribution)
│       ├── logs/                        # Prediction raw logs และ Latency Logs (p50, p95, p99, QPS)
│       ├── graphs/                      # กราฟสรุปผลงานวิจัยขั้นสุดท้าย (Cross-Generator Performance Gap Plots)
│       ├── stat_tests/                  # ผลทดสอบนัยสำคัญทางสถิติ (mcnemar_holm_results.json, wilcoxon_results.json)
│       ├── task4_e2e_results/           # ผลการทดลองพายป์ไลน์รวม Task 4 E2E Pipeline Benchmark
│       └── final_luna_benchmark.csv
```

### 7.2 ชุดกราฟวิเคราะห์ผลลัพธ์ 6 รูปแบบ (6 Standard Visualization Graphs)

1. **Hyperparameter Optimization History Plot (`01_optimization_history.png`):** Scatter & Line Plot ดูพัฒนาการ F1, F2-Score หรือ MAE ในแต่ละ Trial
2. **Model Performance Comparison Bar Chart (`02_model_performance_comparison.png`):** Grouped Bar Chart แสดง F1, F2, Precision, Recall, MCC, MAE เทียบกับ Baselines
3. **F1 & F2-Score vs Latency Trade-Off Scatter Plot (`03_f1_f2_vs_latency_tradeoff.png`):** 2D Scatter Plot แสดง p95 Latency ทั้งแบบ GPU และ CPU
4. **Confusion Matrix & Residual Error Plots (`04_confusion_matrix_heatmap.png`):** Heatmap สำหรับ Classification และ Residual Plot สำหรับ Count Regression
5. **Feature Importance / Top N-gram Weights (`05_feature_importance.png`):** Horizontal Bar Chart แสดง Top 20 Words/Char N-grams
6. **Cross-Generator Performance Gap Comparison Chart (`06_gemini_val_vs_luna_test.png`):** Side-by-Side Bar Chart เปรียบเทียบ F1/F2/MAE ระหว่าง Gemini 5-Fold CV ($\text{Mean} \pm \text{Std}$) กับ Luna Test เพื่อแสดงช่องว่างประสิทธิภาพ (Performance Gap) จาก Domain Shift

### 7.3 ระเบียบวิธีการทดสอบนัยสำคัญทางสถิติที่รัดกุม (Rigorous Statistical Testing Protocol)
1. **การรายงานผล 5-Fold Cross-Validation:** รายงานผลในรูปแบบ **$\text{Mean} \pm \text{Std}$** ข้าม 5 Folds ทั้งคอลัมน์ F1-Score และ F2-Score
2. **McNemar's Test บน Held-Out Luna Test Set:** ใช้ McNemar's Test ในการเปรียบเทียบผลการจำแนกประเภทข้ามข้อความในชุดทดสอบ Luna
3. **Holm-Bonferroni Multiple Comparison Correction:** ปรับแก้ค่า $p$-value ด้วย **Holm-Bonferroni Method** ($p_{\text{adj}} < 0.05$) เมื่อเปรียบเทียบข้ามคู่โมเดล Top-3
4. **Wilcoxon Signed-Rank Test สำหรับตัววัดความผิดพลาดต่อเนื่องรายข้อความ (Residuals):** ใช้ Wilcoxon Signed-Rank Test **เฉพาะกับค่าความผิดพลาดต่อเนื่องรายข้อความทดสอบ (Sample-wise Absolute Error / MAE Residuals)** ใน Task 2 Count Regression ข้ามโมเดล (*ไม่ใช้ Wilcoxon Test กับ Latency*)
5. **การวัด Latency แบบ Empirical Distribution:** รายงานผล Latency และ Throughput (QPS) ในรูป **Empirical Distribution (Mean, p50, p95, p99, QPS)** จากการรัน $N=1,000$ ครั้งโดยไม่แกล้งทำ Hypothesis Testing
6. **เกณฑ์สรุปนัยสำคัญ:** สรุปว่าโมเดลมีความแตกต่างกันอย่างมีนัยสำคัญทางสถิติก็ต่อเมื่อ **$p_{\text{adj}} < 0.05$**

### 7.4 เกณฑ์การตัดสินเลือกโมเดลที่ดีที่สุด (Multi-Criteria Selection Guidelines)

1. **ทั้ง F1-Score และ F2-Score (รวมถึง MAE) บน Luna Test สูงที่สุดอย่างมีนัยสำคัญทางสถิติ ($p_{\text{adj}} < 0.05$):** วัดความสมดุลและความปลอดภัยจริง หลังผ่านการปรับแก้ Holm Correction
2. **Cross-Generator Performance Gap ต่ำ ($\Delta F1, \Delta F2 \le 0.05$):** ช่องว่างประสิทธิภาพไม่กว้างจนเกินไปเมื่อทดสอบข้าม Generator ต่างตระกูล
3. **Inference Latency เปรียบเทียบจริง:** สรุปและรายงานผลลัพธ์ p95 Latency และ Throughput (QPS) จริงของแต่ละโมเดลทั้งบน CPU และ GPU เพื่อประกอบการพิจารณาเลือกโมเดลที่เหมาะสมที่สุดตามความต้องการใช้งาน
4. **Low Critical Under-Triage Rate in Task 3 (พร้อม 95% CI):** สำหรับงานคัดกรองผู้ป่วย โมเดลที่ดีที่สุดต้องมีอัตรา **Under-triage (UR) ต่ำที่สุด โดยขอบเขตบนของ 95% Confidence Interval ต้องไม่เกิน 2.0%** ($\text{Upper bound of 95% CI} \le 2.0\%$)

---

## 8. สถาปัตยกรรมสคริปต์ประมวลผลแบบ Modular CLI & Auto-Checkpointing

สคริปต์การรัน `run_classical_ml.py` รองรับการเทรน/ประเมินราย Task (Task 1, 2, 3.1, 3.2, 3.3) และการรันพายป์ไลน์รวม **Task 4 (E2E Pipeline Benchmark)** บน GPU พร้อมระบบ **Auto Checkpointing**:

### 8.1 พารามิเตอร์ของ CLI Command (`argparse`)
* `--task`: เลือก Task ที่ต้องการรัน (`1`, `2`, `3.1`, `3.2`, `3.3`, `4` (E2E Benchmark), หรือ `all`)
* `--model`: เลือกโมเดลที่ต้องการรัน (เช่น `LinearSVC`, `XGBClassifier`, `XGBRegressor`, `LGBMClassifier`, `BiLSTM_CRF`, หรือ `all`)
* `--use_gpu`: เปิดใช้งาน GPU Acceleration (`true`/`false`, Default: `true`)
* `--run_id`: เลือกรอบการทดลอง (`run_01_baseline_default`, `run_02_autotune_hyperparams`, `run_03_final_luna_benchmark`)
* `--n_trials`: จำนวนรอบการสุ่มจูนของ Optuna (Default: 30 trials)
* `--cv_folds`: จำนวน Folds สำหรับ Cross-Validation (Default: 5 folds)
* `--latency_runs`: จำนวนรอบทดสอบ Latency ($N=1000$ runs)
* `--force`: สั่งให้รันใหม่ทับ Log เดิม (ถ้าไม่ใส่ จะข้ามโมเดลที่รันเสร็จแล้วให้อัตโนมัติ)

### 8.2 ตัวอย่างคำสั่งการรันบน Terminal (Usage Examples)

#### 🔹 1. สั่งรันแยกทีละโมเดล (พักคอมพิวเตอร์ระหว่างรันได้)
```bash
# รัน XGBoost Classifier สำหรับ Task 1 (5-Fold Stratified CV Pipeline)
python run_classical_ml.py --task 1 --model XGBClassifier --use_gpu true --run_id run_02_autotune_hyperparams

# รัน XGBoost Regressor สำหรับทำนายตัวเลขนับ Task 2 (5-Fold Standard CV Pipeline)
python run_classical_ml.py --task 2 --model XGBRegressor --use_gpu true --run_id run_02_autotune_hyperparams

# รัน BiLSTM-CRF บน PyTorch GPU สำหรับ Task 3.1 (People Extractor)
python run_classical_ml.py --task 3.1 --model BiLSTM_CRF --use_gpu true --run_id run_02_autotune_hyperparams

# รัน XGBoost สำหรับ Task 3.2 (Pediatric Triage: เด็ก <= 12 ปี)
python run_classical_ml.py --task 3.2 --model XGBClassifier --use_gpu true --run_id run_02_autotune_hyperparams

# รัน LightGBM สำหรับ Task 3.3 (Adult Triage: ผู้ใหญ่ > 12 ปี)
python run_classical_ml.py --task 3.3 --model LGBMClassifier --use_gpu true --run_id run_02_autotune_hyperparams
```

#### 🔹 2. สั่งรันเร่งความเร็วทุกโมเดล (มีระบบข้ามโมเดลที่เสร็จแล้วอัตโนมัติ)
```bash
# หากกด Ctrl+C เพื่อพักคอม เมื่อกลับมารันคำสั่งเดิม สคริปต์จะรันต่อจากโมเดลถัดไปให้อัตโนมัติ
python run_classical_ml.py --task all --model all --use_gpu true --run_id run_02_autotune_hyperparams --n_trials 30 --cv_folds 5
```

#### 🔹 3. สั่งประกอบร่างโมเดลที่ดีที่สุดเพื่อรัน Task 4: End-to-End Pipeline Benchmark บนชุด Luna Test
```bash
python run_classical_ml.py --task 4 --use_gpu true --measure_latency --latency_runs 1000 --run_id run_03_final_luna_benchmark
```

#### 🔹 4. สั่งวาดกราฟสรุปทุก Task และทดสอบนัยสำคัญทางสถิติประจำรอบ
```bash
python run_classical_ml.py --generate_graphs --run_stat_tests --run_id run_03_final_luna_benchmark
```
