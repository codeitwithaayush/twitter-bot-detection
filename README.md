# 🤖 Twitter Bot Detection using Machine Learning & Deep Learning

> Research paper project presented at the **9th International Conference on Computing in Engineering & Technology**
> Department of Information Technology, Dwarkadas J. Sanghvi College of Engineering, Mumbai

---

## 📌 Overview

A real-time, end-to-end web application that detects whether a Twitter account is a **bot or a human** using behavioral and structural profile features — without analyzing tweet content.

Enter a Twitter username → scrape 23 profile features → MLP inference → **BOT / HUMAN verdict with confidence score** in under 3 seconds.

---

## 🏗️ System Architecture

User (Username Input)
↓
React Frontend
↓
Flask / FastAPI Backend
↓
BrightData Scraper API → 23 Features Extracted
↓
Preprocessing (Log-scale, MinMaxScaler, Ratio-capping)
↓
MLP Model (85ms CPU inference)
↓
BOT / HUMAN + Confidence Score + Feature Rankings


---

## ⚙️ Model Architecture

Input(23) → Dense(128, ReLU, 30% Dropout) → Dense(64, ReLU, 30% Dropout) → Output(1, Sigmoid)


- **Framework:** PyTorch
- **Optimizer:** Adam (lr=0.001, L2 decay=1e-5)
- **Loss:** Binary Cross-Entropy
- **Epochs:** 100 (early stopping, patience=15)
- **No GPU required** — CPU-only inference

---

## 📊 Datasets Used

| Dataset | Size |
|---------|------|
| MGTAB Wastewater | 10,199 accounts |
| Cresci-2017 | 14,368 bots |
| BotOrNot | 15,000 labelled accounts |

Train / Validation / Test split: **7:2:1**

---

## 🧠 Features (23 Total)

| Category | Features |
|----------|----------|
| **Core Counts** | followers_count, friends_count, statuses_count, favourites_count, listed_count |
| **Ratios (Derived)** | followers_friends_ratio (capped at 10), statuses_per_day |
| **Temporal (Derived)** | account_age_days = (current\_date − created\_at) / 10000 |
| **Boolean** | verified, default_profile, geo_enabled, has_url |
| **Profile** | name_length, screen_name_length, description_length (log-scaled) |

### 🏆 Top 5 Discriminative Features (by Info Gain)

| Rank | Feature | Description |
|------|---------|-------------|
| 1 | followers_friends_ratio | Followers/friends capped at 10 |
| 2 | default_profile | Default profile image/settings |
| 3 | listed_count | Public lists containing account |
| 4 | description_length | Bio length (log-scaled) |
| 5 | account_age_days | Days since creation (normalized) |

---

## 📈 Results

### MLP Model Performance
| Metric | Value |
|--------|-------|
| Test Accuracy | 82.39% |
| Validation Accuracy | 93% |
| Precision | 91% |
| Recall | 94% |
| F1-Score | 0.925 |
| CPU Inference | 85ms |
| End-to-End Latency | < 3 seconds |

### Model Comparison
| Model | Accuracy | F1-Score |
|-------|----------|----------|
| XGBoost | 98.25% | 0.9875 |
| Random Forest | 98.21% | 0.9872 |
| Gradient Boosting | 98.13% | 0.9866 |
| **MLP (Ours)** | **82.39%** | **0.7088** |
| SVM | 93.65% | 0.9556 |
| Logistic Regression | 90.63% | 0.9366 |

> ⚡ MLP chosen over XGBoost for **3.5× faster inference (85ms vs 300ms+)** and **no GPU dependency** — critical for real-time deployment.

### Comparison with State-of-the-Art Systems
| System | Accuracy | Inference | Hardware | Features |
|--------|----------|-----------|----------|----------|
| Botometer | 94% | 2–5s | Cloud | 1,200+ |
| DeeProBot | 92% | 450ms | GPU | Profile |
| TL-PBot | 98% | 300ms | GPU | Profile |
| **MLP (Ours)** | **82%** | **85ms** | **CPU** | **23** |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Backend
```bash
cd backend
python app.py
```

### Run the Frontend
```bash
cd frontend
npm install
npm start
```

### Usage
1. Open the web app in your browser
2. Enter any Twitter/X username
3. Get instant **BOT / HUMAN** verdict with:
   - Confidence score (out of 100)
   - Feature importance breakdown
   - Processing latency

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React.js |
| Backend | Flask / FastAPI |
| ML Model | PyTorch (MLP) |
| Data Scraping | BrightData API |
| Preprocessing | Scikit-learn (MinMaxScaler) |

---

## ⚠️ Limitations

- Low-activity human accounts may be incorrectly flagged as bots
- Private/protected accounts return `UNANALYZABLE`
- No NLP on tweet content (by design — resistant to LLM-generated text)
- Accuracy may vary with highly adaptive bots

---

## 🔮 Future Work

- Integrate graph-based features (follower network analysis)
- Add NLP for AI-generated tweet detection
- Transfer learning across platforms
- Continuous/online learning as bot strategies evolve
- Explainable AI (XAI) for moderation transparency

---

## 📄 Citation

If you use this work, please cite:

Akshit Soji, Ashutosh Shetty, Aditya Singh, Aayush Trada, Pravin Hole, Prachi Satam, Satishkumar Varma.
"Bot Detection for Twitter Accounts using Machine Learning and Deep Learning Models."
9th International Conference on Computing in Engineering & Technology (ICCET), 2025.
Dwarkadas J. Sanghvi College of Engineering, Mumbai, India.


---

## 📜 License

This project is for academic and research purposes only.
