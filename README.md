# E-Commerce Customer Churn Predictor

This project is a web application that predicts the probability of customer churn for an e-commerce business using a machine learning model (Random Forest). It provides both an interactive web interface and a REST API for real-time predictions.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Model & Algorithm](#model--algorithm)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [API Usage](#api-usage)
- [Project Structure](#project-structure)
- [License](#license)

---

## Features

- Predicts customer churn probability based on various customer metrics
- Interactive and responsive web interface
- Real-time predictions via web and API
- Visual representation of churn probability
- Handles missing data gracefully
- Cloud deployment ready (e.g., Render)

---

## Tech Stack

- **Language:** Python
- **Web Framework:** Flask
- **Machine Learning:** scikit-learn (RandomForestClassifier)
- **Data Processing:** pandas, numpy
- **Model Serialization:** joblib
- **Deployment:** Render (cloud platform)

---

## Dataset

- **Name:** `E Commerce Dataset.csv`
- **Description:** Contains customer demographic, behavioral, and transactional data for an e-commerce platform.
- **Features:**
  - **Categorical:** PreferredLoginDevice, PreferredPaymentMode, Gender, PreferedOrderCat, MaritalStatus
  - **Numerical:** Tenure, CityTier, WarehouseToHome, HourSpendOnApp, NumberOfDeviceRegistered, SatisfactionScore, NumberOfAddress, Complain, OrderAmountHikeFromlastYear, CouponUsed, OrderCount, DaySinceLastOrder, CashbackAmount
  - **Target:** Churn (1 = churned, 0 = active)
- **Note:** The dataset must be placed in the project directory before training.

---

## Model & Algorithm

- **Model:** RandomForestClassifier from scikit-learn
- **Algorithm:** Random Forest (ensemble of decision trees)
- **Why Random Forest?**
  - Robust to overfitting
  - Handles mixed data types
  - Provides feature importance
  - Suitable for tabular business data

---

## How It Works

1. **Data Loading:** Loads `E Commerce Dataset.csv` for training. If missing, creates a dummy model.
2. **Preprocessing:** Encodes categorical features, imputes and scales numerical features.
3. **Model Training:** Trains a RandomForestClassifier on the processed data.
4. **Model Saving:** Saves the model and preprocessors as `.joblib` files.
5. **Prediction:** Accepts new customer data, preprocesses it, and predicts churn probability.
6. **Web/API Interface:** Users can interact via a web form or send JSON to the `/predict` API endpoint.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Place your dataset:**
   - Add `E Commerce Dataset.csv` to the project root directory.

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application:**
   ```bash
   python app.py
   ```

---

## Usage

- Open your browser and go to `http://localhost:5000`
- Fill in the customer details in the web form and submit to see the churn probability.

---

## API Usage

- **Endpoint:** `/predict`
- **Method:** POST
- **Content-Type:** application/json

**Sample Request:**
```json
{
  "PreferredLoginDevice": "Mobile Phone",
  "PreferredPaymentMode": "Credit Card",
  "Gender": "Male",
  "PreferedOrderCat": "Fashion",
  "MaritalStatus": "Single",
  "Tenure": 12,
  "CityTier": 2,
  "WarehouseToHome": 5,
  "HourSpendOnApp": 1.5,
  "NumberOfDeviceRegistered": 3,
  "SatisfactionScore": 4,
  "NumberOfAddress": 2,
  "Complain": 0,
  "OrderAmountHikeFromlastYear": 10.5,
  "CouponUsed": 2,
  "OrderCount": 15,
  "DaySinceLastOrder": 30,
  "CashbackAmount": 50.0
}
```

**Sample Response:**
```json
{
  "success": true,
  "churn_probability": 23.45
}
```

---

## Project Structure

```
.
├── app.py
├── requirements.txt
├── render.yaml
├── .gitignore
├── E Commerce Dataset.csv
├── model.joblib
├── scaler.joblib
├── imputer.joblib
├── label_encoders.joblib
├── features.joblib
├── templates/
│   └── index.html
├── static/
│   └── favicon.ico
└── README.md
```

---

## License

This project is licensed under the MIT License.

---

**Note:**  
- Make sure to keep your dataset and model files secure and do not upload sensitive data to public repositories.
- For deployment on Render or other cloud platforms, ensure all environment variables and configuration files are set
