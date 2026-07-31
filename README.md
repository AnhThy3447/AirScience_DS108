# AIRSCIENCE: Airline Pricing Analysis & Recommendation System

## Table of Contents
- [1. Overview](#1-overview)
- [2. Features](#2-features)
- [3. Dataset](#3-dataset)
- [4. Project Structure](#4-project-structure)
- [5. Execution](#5-execution)
- [6. Related Documents](#6-related-documents)
- [7. Team Members](#7-team-members)

## 1. Overview
The team collected and analyzed flight ticket pricing data, then applied machine learning techniques to build a system that predicts ticket prices over time. The goal of this system is to recommend the optimal time to purchase a ticket, helping customers make smarter, more cost-effective decisions.

## 2. Features
- Analyzes the data to uncover hidden relationships between ticket price and other flight attributes.
- Uses machine learning models to recommend the best time to book a flight.
- Takes basic flight ticket attributes as input and outputs a predicted price, which is then used to determine a reasonable booking time and a "buy now vs. wait" recommendation.
- A demo website provides booking-time recommendations to help users choose the right time to purchase within their budget, tailored to the Vietnamese market.

## 3. Dataset

Data is organized using a tiered **Medallion Architecture**.

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
- Data has been cleaned, missing values handled, and formats standardized.
- Size: 46,549 samples with 11 features.

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
Data standardized and segmented by airline, ready for analysis.

## 4. Project Structure
```bash
.
├── Source_code
    ├── Data
    │   ├── Bronze_layer/
    │   │   └── merged_file.csv           
    │   ├── Silver_layer/
    │   │   └── cleaned_file.csv          
    │   ├── Gold_layer/
    │   │   ├── Bamboo_Airways.csv
    │   │   ├── VietJet_Air.csv
    │   │   ├── Vietnam_Airlines.csv
    │   │   └── Vietravel_Airlines.csv   
    │
    ├── 01_crawling.ipynb                 # Data collection
    ├── 02_preprocessing.ipynb            # Data preprocessing: cleaning, formatting, handling missing values
    ├── 03_EDA.ipynb                      # Exploratory Data Analysis
    ├── 04_modeling.ipynb                 # Model training and evaluation
    ├── 05_web_demo                       # Interactive demo website
├── paper.pdf
├── slides.pdf
├── README.md                         # Project description file
```

## 5. Execution
### Data Collection:
- Used Selenium combined with Microsoft Edge for web scraping.
- Limitation: Not fully automated due to Traveloka's CAPTCHA blocking. 
### Data Exploration:
Based on the EDA results, the team took extra care in handling outliers. The team also decided to split the data by airline rather than by destination or other factors.
    ![](https://github.com/user-attachments/assets/4a3a8c6d-299a-43f4-ab5d-65169c4444e8)
    ![](https://github.com/user-attachments/assets/8a4ca0ab-2f91-4d7a-bcea-b7c4027f6b85)

### Model Training:
- Data split:
    - `Test_data`: 40 flight IDs
    - `Train + Test`:
        - `Test` = 18% * (dataset - Test_data) + Test_data
        - `Train` = dataset - Test
    - Outliers were handled separately for each file.

- Model training (`Train + Test`):
    - Metrics: `R2-Score`, `MAE`, `MAPE`
    - Models evaluated: `AdaBoost`, `BaggingRegressor`, `GradientBoostingRegressor`, `DecisionTreeRegressor`, `RandomForestRegressor`, `ExtraTreesRegressor`
    - Selected model: **Gradient Boosting Regressor**, whose predicted price trends for Bamboo Airways, Vietnam Airlines, VietJet Air, and Vietravel Airlines closely matched actual values.
    ![](https://github.com/user-attachments/assets/d43509aa-bef7-47f4-917f-60f6f176fadc)

- Experimentation (`Test_data`):

    To apply the model in practice, the team used data from 40 flights to determine an appropriate purchase threshold, from which a **Buy (1)** or **Wait (0)** recommendation is generated.

    Threshold calculation process:
    - Step 1: Predict ticket prices for upcoming days using the trained Gradient Boosting Regressor model:

        `P={p_1,p_2,p_3,…,p_n }`
    - Step 2: Select the lowest 10% of predicted prices (k = number of tickets):

        `P_low={p_1,p_2,p_3,…,p_k }`
    - Step 3: Compute the threshold as the average of P_low:

        `threshold=1/k ∑_(i=1)^k p_i`

    Based on the threshold, the system generates a recommendation:
    - Left image: When the ticket price is higher than the threshold, the model recommends waiting (label = 0).
    - Right image: When the ticket price is lower than the threshold, the model recommends buying (label = 1).
    ![](https://github.com/user-attachments/assets/183ba274-acd3-4020-9353-99f9f081bc52)

### Running the Demo Website:
![](https://github.com/user-attachments/assets/88993a36-6209-42e0-af43-f21339476ee8)
![](https://github.com/user-attachments/assets/9e24ad66-906b-4e52-822a-c7b66ffa4384)

## 6. Related Documents
- Detailed report: [(paper.pdf)](paper.pdf)  
- Presentation slides: [(slides.pdf)](slides.pdf)

## 7. Team Members
| Name                    | Student ID |
|-------------------------|------------|
| Đinh Bảo Thy            | 23521563   |
| Võ Ngọc Anh Thy         | 23521565   |
| Nguyễn Vũ Thùy Trâm     | 23521617   |
