# 🌫️ AQI Predictor — Air Quality Index Estimation & Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Visuals-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning and Data Analytics project predicting the **National Air Quality Index (AQI)** for major Indian cities using atmospheric pollutant concentrations and temporal trends. Deployed as a web application on Streamlit.

---

## 📌 Project Overview

- **Author:** Aniket Kanojiya  
- **Club / Program:** 1M1B - USAR  
- **GitHub Repository:** [https://github.com/Aniket-Kanojiya/AQI-Predictor](https://github.com/Aniket-Kanojiya/AQI-Predictor)  
- **Live Web Application:** [Add your Streamlit Cloud link here]  
- **Official Documentation:** [📄 Project Report (Markdown)](./PROJECT_REPORT.md)  

---

## 🎯 UN Sustainable Development Goals (SDG) Alignment

This initiative directly aligns with the United Nations Sustainable Development Goals:

- **SDG 3: Good Health and Well-Being (Target 3.9):** Reduces morbidity and mortality caused by hazardous air pollutants by delivering real-time toxicity feedback and preventative health categories.
- **SDG 11: Sustainable Cities and Communities (Target 11.6):** Helps monitor and understand urban air quality dynamics across major metropolitan hubs.
- **SDG 13: Climate Action (Target 13.3):** Promotes data-driven environmental education and climate mitigation awareness.

---

## 🏗️ System Architecture

`
                       USER
   (City selection, pollutant sliders, or historical date)
                         │
                         ▼
             STREAMLIT USER INTERFACE
                         │
                         ▼
        INPUT PREPROCESSING & FEATURE ENCODING
  (Temporal extraction + Season mapping + One-Hot City + Reindexing)
                         │
                         ▼
        TRAINED REGRESSION MODEL (aqi_model.pkl)
                         │
                         ▼
             ESTIMATED NUMERICAL AQI (0–500)
                         │
                         ▼
  NAQI HEALTH CATEGORY | REAL-TIME GAUGE | 30-DAY ROLLING TRENDS
                         │
                         ▼
    HISTORICAL RECORD COMPARISON & MODEL METRICS
`

---

## ✨ Key Features

- 🏙️ **Multi-City Scope:** Real-time simulations and analysis for **Delhi, Mumbai, Kolkata, Chennai, and Bangalore**.
- 🎛️ **Dual Operating Modes:**
  - **Manual Simulation:** Adjust sliders for 12 criteria pollutants (PM2.5, PM10, NO2, SO2, CO, O3, etc.) to predict instant AQI.
  - **Historical Lookup:** Select any historical date (2015–2024) to inspect actual sensor recordings vs model predictions.
- 🚦 **NAQI Categorization:** Automated mapping to official Indian National Air Quality Index color buckets (Good, Satisfactory, Moderate, Poor, Very Poor, Severe).
- 📈 **30-Day Rolling Trends:** Multi-select interactive Plotly comparison overlaying historical pollution trajectories with standard risk bands.
- ⚡ **Lightweight Architecture:** Zero-bloat design following the **Ponytail** senior developer principles, ensuring minimal dependencies, fast inference, and seamless Streamlit Community Cloud hosting.

---

## 📊 Dataset & NAQI Standards

Data originates from the Central Pollution Control Board (CPCB) monitoring network (2015–2024), containing ~18,265 daily records across 5 metropolitan centers.

### Indian National Air Quality Index (NAQI) Scale

| AQI Range | Category | Color Hex | Health Advisory |
| :---: | :---: | :---: | :--- |
| **0 – 50** | **Good** | #2ecc71 | Minimal impact |
| **51 – 100** | **Satisfactory** | #a3d977 | Minor breathing discomfort to sensitive people |
| **101 – 200** | **Moderate** | #f1c40f | Breathing discomfort with lung/heart disease |
| **201 – 300** | **Poor** | #e67e22 | Breathing discomfort to most people on prolonged exposure |
| **301 – 400** | **Very Poor** | #e74c3c | Respiratory illness on prolonged exposure |
| **401 – 500** | **Severe** | #8b0000 | Affects healthy people, seriously impacts vulnerable |

---

## 📁 Repository Structure

`
AQI-Predictor/
├── app/
│   └── app.py                 # Streamlit web application
├── data/
│   ├── city_day.csv           # Raw CPCB monitoring dataset
│   ├── cleaned_aqi.csv        # Cleaned dataset (outliers capped, imputed)
│   └── features.csv           # Feature engineered dataset (time, lag features)
├── models/
│   ├── aqi_model.pkl          # Trained regression model
│   ├── feature_columns.pkl    # Serialized feature column list
│   └── metrics.json           # Model evaluation metrics
├── reports/
│   ├── figures/               # Generated exploratory data analysis plots
│   └── PROJECT_REPORT.md      # Detailed academic / institutional project report
├── src/
│   ├── data_cleaning.py       # Data validation, outlier capping & NAQI buckets
│   ├── eda.py                 # Generates 5 EDA visual plots
│   ├── features.py            # Temporal & lag feature extraction
│   └── train.py               # Model training & serialization
├── .gitignore                 # Excludes cache and non-essential binaries
├── AGENTS.md                  # Development principles (Ponytail lazy senior dev mode)
├── PROJECT_REPORT.md          # Root project report reference
├── requirements.txt           # Production dependencies
└── README.md                  # Project documentation
`

---

## 🚀 Getting Started

### 1. Clone the Repository
`ash
git clone https://github.com/Aniket-Kanojiya/AQI-Predictor.git
cd AQI-Predictor
`

### 2. Set Up Virtual Environment
`ash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS / Linux
python3 -m venv venv
source venv/bin/activate
`

### 3. Install Dependencies
`ash
pip install -r requirements.txt
`

### 4. Run the Pipeline (Optional)
To regenerate cleaned data, exploratory figures, and model weights:
`ash
python src/data_cleaning.py
python src/eda.py
python src/features.py
python src/train.py
`

### 5. Launch the Streamlit App
`ash
streamlit run app/app.py
`
Visit **http://localhost:8501** in your web browser.

---

## 📈 Exploratory Data Analysis & Visuals

The automated EDA suite in src/eda.py outputs high-resolution figures saved in eports/figures/:
1. qi_trends.png — Multi-year monthly average AQI trends per city
2. correlation_heatmap.png — Inter-pollutant correlation matrix
3. seasonal_patterns.png — Distribution of AQI across Indian seasons (Winter, Summer, Monsoon, Post-Monsoon)
4. pm25_vs_aqi.png — Scatter analysis highlighting the strong influence of fine particulate matter
5. missing_data.png — Zero-value and data sparsity distributions

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Acknowledgments & Contact

- **Dataset:** Central Pollution Control Board (CPCB), India via Kaggle.
- **Developer:** [Aniket Kanojiya](https://github.com/Aniket-Kanojiya)
- **Organization:** 1M1B (One Million for One Billion) Foundation - USAR
