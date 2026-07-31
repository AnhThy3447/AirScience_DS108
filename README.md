# AirScience: Airline Pricing Analysis & Recommendation System

---

## 📌 Table of Contents
- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Data Design (Medallion Architecture)](#-data-design-medallion-architecture)
- [Data Pipeline](#-data-pipeline)
- [Modeling & Analysis](#-modeling--analysis)
- [Results & Recommendation System](#-results--recommendation-system)
- [Tech Stack](#-tech-stack)
- [Project Setup](#-project-setup)
- [Contributors](#-contributors)
- [License](#-license)

---

## 📖 Overview
AirScience is an end-to-end system for collecting and analyzing airline ticket pricing data in the Vietnamese market. The project applies machine learning to build a price-prediction model that forecasts how a flight's price will move over time, and uses that forecast to recommend the optimal moment to buy — helping travelers avoid overpaying and make informed booking decisions. The final output is a demo website that surfaces this recommendation directly to users.

---

## 🏗 System Architecture
The pipeline is designed across 4 key stages:
1. **Data Collection:** Scraped flight ticket listings from Traveloka using Selenium, sampling each flight at multiple points in the booking window.
2. **Data Preprocessing:** Cleaned raw data, handled missing values, standardized formats, and engineered features (organized via a Bronze → Silver → Gold layered structure).
3. **Modeling:** Trained and compared multiple regression models to predict ticket price as a function of time-to-departure and other flight attributes.
4. **Recommendation Engine & Demo:** Converted price predictions into a binary "Buy now" / "Wait" recommendation, and served it through an interactive demo website.

---

## 📐 Data Design (Medallion Architecture)
Data is organized using a tiered **Medallion Architecture** (Bronze / Silver / Gold) to progressively clean, standardize, and segment the dataset for analysis.

### Bronze Layer
Raw data collected from the Traveloka website, including:
- `id` (assigned by the team): flight ID, with each flight having at least 2 samples corresponding to 2 different purchase dates.
- `start_day`, `end_day`, `crawl_date`: `crawl_date` ranges from April 7, 2025 to May 10, 2025, and is always 20 days before `start_day`.
- `destination`:
    - Major cities: Hanoi (HAN), Hai Phong (HPH), Da Nang (DAD)
    - Tourist destinations: Phu Quoc (PQC), Nha Trang (CXR), Da Lat (DLI)
- `brand` (airline, determined based on flights to the predefined destinations): Bamboo Airways, Vietnam Airlines, VietJet Air, Vietravel Airlines
- `price`, `start_time`, `end_time`, `trip_time`, `checked_baggage`, `hand_luggage`

### Silver Layer
Cleaned data with missing values handled and formats standardized — **46,549 samples across 11 features**:

| Index | Feature           | Description                       | Type         |
|-------|-------------------|------------------------------------|--------------|
| 1     | `id`              | Flight ID                          | categorical  |
| 2     | `brand`           | Airline                            | categorical  |
| 3     | `price`           | Ticket price (VND/passenger)       | numeric      |
| 4     | `destination`     | Destination                        | categorical  |
| 5     | `hand_luggage`    | Hand luggage allowance (kg)        | numeric      |
| 6     | `checked_baggage` | Checked baggage allowance (kg)     | numeric      |
| 7     | `start_hour`      | Departure time slot                | categorical  |
| 8     | `end_hour`        | Arrival time slot                  | categorical  |
| 9     | `trip_mins`       | Flight duration                    | numeric      |
| 10    | `is_holidays`     | Holiday classification of the day  | categorical  |
| 11    | `days_left`       | Days remaining before departure    | numeric      |

### Gold Layer
Data standardized and segmented by airline (`Bamboo_Airways.csv`, `VietJet_Air.csv`, `Vietnam_Airlines.csv`, `Vietravel_Airlines.csv`), ready for modeling.

---

## 🔄 Data Pipeline
- **Collection:** Selenium + Microsoft Edge automation against Traveloka. *Limitation: not fully automated due to CAPTCHA blocking*, requiring some manual intervention during scraping.
- **Exploratory Data Analysis:** Careful handling of outliers was required; the team ultimately split the dataset **by airline** rather than by destination or other candidate factors, based on EDA findings.

<img src="https://github.com/user-attachments/assets/4a3a8c6d-299a-43f4-ab5d-65169c4444e8" width="800" alt="EDA distribution" />
<img src="https://github.com/user-attachments/assets/8a4ca0ab-2f91-4d7a-bcea-b7c4027f6b85" width="800" alt="EDA relationships" />

---

## 📊 Modeling & Analysis

### Data Split
- `Test_data`: 40 flight IDs, held out for the final recommendation experiment.
- `Train + Test`:
  - `Test` = 18% × (dataset − `Test_data`) + `Test_data`
  - `Train` = dataset − `Test`
- Outliers were handled separately per airline file.

### Model Comparison
Six regression models were trained and evaluated using `R²-Score`, `MAE`, and `MAPE`:
`AdaBoost`, `BaggingRegressor`, `GradientBoostingRegressor`, `DecisionTreeRegressor`, `RandomForestRegressor`, `ExtraTreesRegressor`

**Selected model: Gradient Boosting Regressor** — its predicted price trends for Bamboo Airways, Vietnam Airlines, VietJet Air, and Vietravel Airlines closely tracked actual observed prices.

<img src="https://github.com/user-attachments/assets/d43509aa-bef7-47f4-917f-60f6f176fadc" width="800" alt="Model performance comparison" />

---

## 🎯 Results & Recommendation System
Using the trained model on the held-out `Test_data` (40 flights), the team derived a purchase threshold to convert price forecasts into a simple **Buy / Wait** signal:

1. **Predict** future prices for a flight across the remaining booking window: `P = {p_1, p_2, …, p_n}`
2. **Select** the lowest 10% of predicted prices (`k` tickets): `P_low = {p_1, p_2, …, p_k}`
3. **Compute** the threshold as the mean of `P_low`: `threshold = (1/k) Σ p_i`

The system then recommends:
- **Wait (label = 0)** — when the current price is above the threshold.
- **Buy (label = 1)** — when the current price is at or below the threshold.

<img src="https://github.com/user-attachments/assets/183ba274-acd3-4020-9353-99f9f081bc52" width="800" alt="Buy vs Wait recommendation" />

### Demo Website
The recommendation system is served through an interactive demo website where users can check the buy/wait signal for a given flight.

<img src="https://github.com/user-attachments/assets/88993a36-6209-42e0-af43-f21339476ee8" width="800" alt="Demo website" />
<img src="https://github.com/user-attachments/assets/9e24ad66-906b-4e52-822a-c7b66ffa4384" width="800" alt="Demo website result" />

---

## 🛠 Tech Stack
* **Data Collection:** Selenium, Microsoft Edge WebDriver
* **Data Processing:** Python (`pandas`, `numpy`)
* **Modeling:** scikit-learn (`GradientBoostingRegressor` and other ensemble regressors)
* **Web Demo:** *(add your framework here, e.g. Flask / Streamlit / React)*

---

## ⚙️ Project Setup

1. **Clone Repository:**
   ```bash
   git clone https://github.com/your-username/airscience.git
   cd airscience
   ```
2. **Data Collection:**
   ```bash
   jupyter notebook Source_code/01_crawling.ipynb
   ```
3. **Preprocessing:**
   ```bash
   jupyter notebook Source_code/02_preprocessing.ipynb
   ```
4. **Exploratory Data Analysis:**
   ```bash
   jupyter notebook Source_code/03_EDA.ipynb
   ```
5. **Model Training:**
   ```bash
   jupyter notebook Source_code/04_modeling.ipynb
   ```
6. **Run the Demo Website:**
   ```bash
   cd Source_code/05_web_demo
   # add your run command here, e.g. python app.py
   ```

---

## 📄 Related Documents
- Detailed report: [paper.pdf](paper.pdf)
- Presentation slides: [slides.pdf](slides.pdf)

---

## 👥 Contributors

| Student ID | Name | Role & Core Contributions |
| :--- | :--- | :--- |
| **23521563** | **Đinh Bảo Thy** | *(add specific contributions)* |
| **23521565** | **Võ Ngọc Anh Thy** | *(add specific contributions)* |
| **23521617** | **Nguyễn Vũ Thùy Trâm** | *(add specific contributions)* |

---
📜 License
This repository is an academic project created for coursework purposes under course **DS108 - Data Collection and Preprocessing** at the **University of Information Technology (UIT), VNU-HCM**. All rights reserved by the project authors.
