# AQI PREDICTOR
## AI/ML-Based Air Quality Index Estimation & Analytics Platform for Indian Cities
### PROJECT REPORT

---

**Submitted By:**  
Aniket Kanojiya  
*(Student ID / Roll No.)*  

**Club / Organization:**  
1M1B - USAR  

**Date:**  
04.09.2026  

**Live Project:**  
*https://aqi-predictor-india.streamlit.app/ (To be updated upon deployment)*  

**GitHub Repository:**  
[https://github.com/Aniket-Kanojiya/AQI-Predictor](https://github.com/Aniket-Kanojiya/AQI-Predictor)  

---

## 1. Problem Statement

Air pollution is among the most severe environmental and public health hazards facing urban India. Major metropolitan centers—including Delhi, Mumbai, Kolkata, Chennai, and Bangalore—consistently report particulate matter and toxic gaseous pollutant concentrations that exceed safety thresholds established by the World Health Organization (WHO) and the Central Pollution Control Board (CPCB).

The **Air Quality Index (AQI)** serves as a standardized indicator to communicate how polluted the ambient air currently is and what associated health effects might be of concern. However, AQI is determined through complex multi-pollutant non-linear sub-index calculations based on 24-hour monitored values of PM2.5, PM10, NO2, NH3, SO2, CO, and O3. 

For citizens, urban planners, and environmental researchers:
- Raw pollutant concentrations recorded at environmental stations are difficult to interpret without dedicated computation tools.
- Existing government portals often report delayed historical indices rather than real-time simulated forecasts.
- There is a lack of accessible, interactive tools that allow users to simulate how altering individual pollutant levels influences overall air toxicity.

This project addresses these challenges by developing a machine-learning-driven analytics and estimation platform that accurately predicts the Air Quality Index and maps it directly to health risk categories.

---

## 2. Proposed Solution

**AQI Predictor** is an interactive machine-learning-powered platform developed in Python and Streamlit. The application models the non-linear relationship between 12 criteria pollutants, geographical city indicators, temporal and seasonal factors, and the resulting Air Quality Index.

Users interact with the platform through two intuitive operational modes:
1. **Manual Input Simulation:** Users adjust intuitive sliders representing concentrations of individual pollutants (PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene). The platform passes these features through a trained regression model, computes the estimated AQI, and displays an animated gauge color-coded to the official Indian National Air Quality Index (NAQI) bucket.
2. **Historical Date Mode:** Users select a city and any past date between 2015 and 2024 to retrieve actual environmental sensor records and compare ground-truth recorded AQI with model predictions.
3. **Multi-City Comparative Analytics:** An interactive visualizer computes and renders 30-day moving averages across multiple selected cities simultaneously, providing comparative perspective on urban air quality trajectories.

---

## 3. How It Works

| Stage | Description |
| :--- | :--- |
| **1. User Input** | City selection, operating mode choice, and pollutant concentrations or historical dates entered via the Streamlit interface. |
| **2. Feature Engineering & Preprocessing** | Temporal features (month, day_of_week, day_of_year, season) are extracted. City indicators are one-hot encoded and aligned with the trained feature schema via reindexing. |
| **3. Machine Learning Prediction** | The serialized regression model (qi_model.pkl) evaluates the input vector and outputs a continuous numerical AQI estimate. |
| **4. NAQI Health Categorization** | The numerical prediction is mapped to official CPCB health categories (Good, Satisfactory, Moderate, Poor, Very Poor, Severe) with standardized color schemes. |
| **5. Interactive Visualization & Feedback** | Results are rendered through animated Plotly gauge indicators, health advisory banners, prediction error metrics, and 30-day multi-city rolling trend plots. |

---

## 4. Key Features

- **Multi-Pollutant AQI Estimation:** Predicts overall AQI using 12 criteria and chemical pollutants.
- **NAQI-Compliant Classification:** Maps outputs to the 6 official Indian air quality categories (0–50: Good, 51–100: Satisfactory, 101–200: Moderate, 201–300: Poor, 301–400: Very Poor, 401–500: Severe).
- **Interactive Simulation Controls:** Slider-based inputs with pollutant-specific bounds reflecting real-world atmospheric thresholds.
- **Historical Ground-Truth Validation:** Ability to query historical monitoring dates (2015–2024) across 5 major metropolitan cities to observe recorded vs. predicted comparisons.
- **Multi-City Trend Visualizer:** Dynamic Plotly chart displaying 30-day rolling averages with color-coded AQI severity backgrounds.
- **Zero-Boilerplate Lightweight Design:** Optimized pipeline following senior engineering principles (Ponytail methodology) with minimal dependencies and fast inference.
- **Cloud-Ready Architecture:** Designed for one-click deployment on Streamlit Community Cloud.

---

## 5. Tech Stack and Tools

| Technology / Tool | Category | Purpose |
| :--- | :--- | :--- |
| **Python 3.10+** | Programming | Core programming language for data engineering and modeling |
| **Streamlit** | Web Application & UI | Interactive frontend dashboard and responsive web controls |
| **Pandas** | Data Processing | Data manipulation, date grouping, cleaning, and rolling averages |
| **NumPy** | Data Processing | Vectorized numerical operations and data reshaping |
| **Scikit-learn** | Machine Learning | Regression modeling, train-test splitting, and error metrics (RMSE, MAE, R²) |
| **Plotly Express & Graph Objects** | Data Visualization | Interactive gauge indicators, multi-line charts, and background bands |
| **Matplotlib & Seaborn** | Exploratory Data Analysis | Static reporting plots, correlation heatmaps, and distribution figures |
| **Joblib** | Model Serialization | Persistence of trained models and feature column schemas |
| **Git & GitHub** | Version Control | Source code management, issue tracking, and repository hosting |
| **Streamlit Community Cloud** | Deployment | Cloud hosting platform for public web accessibility |

---

## 6. Dataset Information

The project utilizes comprehensive ambient air quality monitoring records spanning **2015 to 2024** across five major Indian metropolitan centers: **Delhi, Mumbai, Kolkata, Chennai, and Bangalore**.

- **Source:** Central Pollution Control Board (CPCB) stations via Kaggle (city_day.csv).
- **Total Records:** 18,265 daily observation rows.
- **Target Variable:** AQI (Continuous numerical target ranging from 0 to 500).

### Monitored Features

| Feature | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| **City** | Categorical | — | Metropolitan city (Delhi, Mumbai, Kolkata, Chennai, Bangalore) |
| **Date** | Datetime | YYYY-MM-DD | Date of observation |
| **PM2.5** | Numerical | µg/m³ | Fine particulate matter (≤ 2.5 µm diameter) |
| **PM10** | Numerical | µg/m³ | Coarse particulate matter (≤ 10 µm diameter) |
| **NO** | Numerical | µg/m³ | Nitric oxide concentration |
| **NO2** | Numerical | µg/m³ | Nitrogen dioxide concentration |
| **NOx** | Numerical | ppb | Total nitrogen oxides concentration |
| **NH3** | Numerical | µg/m³ | Ammonia concentration |
| **CO** | Numerical | mg/m³ | Carbon monoxide concentration |
| **SO2** | Numerical | µg/m³ | Sulfur dioxide concentration |
| **O3** | Numerical | µg/m³ | Ambient ozone concentration |
| **Benzene** | Numerical | µg/m³ | Volatile aromatic hydrocarbon |
| **Toluene** | Numerical | µg/m³ | Industrial solvent hydrocarbon |
| **Xylene** | Numerical | µg/m³ | Chemical solvent pollutant |
| **AQI** | Numerical | Index (0–500) | **Target Variable:** National Air Quality Index |
| **AQI_Bucket** | Categorical | NAQI Level | Health category (Good, Satisfactory, Moderate, Poor, Very Poor, Severe) |

### Preprocessing & Quality Engineering:
- **Date Handling & Sorting:** Parsed chronological timestamps and sorted by [City, Date].
- **Deduplication:** Dropped duplicate observations preserving first-reported station data.
- **Outlier Capping:** Applied 99.5th percentile per-city capping on pollutant extremes to protect against sensor anomalies.
- **Missing Value Imputation:** Applied time-series linear interpolation (limit=3) followed by per-city median imputation.
- **Feature Engineering:** Extracted temporal indicators (month, day_of_week, day_of_year, season), per-city lag features (AQI_lag_1, AQI_lag_7, PM25_lag_1), and one-hot encoded city designations.

---

## 7. System Architecture

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

## 8. SDG Relevance

AQI Predictor directly supports the **United Nations Sustainable Development Goals (UN SDGs)**:

### 🎯 Primary Alignment: SDG 3 — Good Health and Well-Being
- **Target 3.9:** *By 2030, substantially reduce the number of deaths and illnesses from hazardous chemicals and air, water and soil pollution and contamination.*
- **Contribution:** Air pollution is a leading contributor to stroke, heart disease, lung cancer, and acute respiratory illnesses in India. By providing accessible, instant assessments of air toxicity levels and corresponding health danger buckets, the platform enables vulnerable populations (children, elderly, asthmatics) to take preventive action.

### 🏙️ Secondary Alignment: SDG 11 — Sustainable Cities and Communities
- **Target 11.6:** *By 2030, reduce the adverse per capita environmental impact of cities, including by paying special attention to air quality and municipal and other waste management.*
- **Contribution:** Offers multi-year comparative trend data across Indian urban hubs, assisting researchers and citizens in monitoring seasonal patterns and urban emission trajectories.

### 🌍 Tertiary Alignment: SDG 13 — Climate Action
- **Target 13.3:** *Improve education, awareness-raising and human and institutional capacity on climate change mitigation, adaptation, impact reduction and early warning.*
- **Contribution:** Demonstrates data-driven tools that bridge the gap between technical environmental monitoring and community-level awareness.

> **Disclaimer:** This software is developed for educational, analytical, and portfolio purposes. It does not replace statutory environmental health bulletins or emergency warnings issued by the Central Pollution Control Board (CPCB) or state pollution control bodies.

---

## 9. Project Links

- **Live Application:** [Add Live URL here — e.g., https://aqi-predictor-india.streamlit.app/]
- **GitHub Repository:** [https://github.com/Aniket-Kanojiya/AQI-Predictor](https://github.com/Aniket-Kanojiya/AQI-Predictor)

---
*Report generated for 1M1B - USAR AI/ML Project Portfolio • September 2026*
