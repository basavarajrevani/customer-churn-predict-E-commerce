# E-Commerce Customer Churn Predictor

This is a web application that predicts the probability of customer churn for an e-commerce business using machine learning.

## Features

- Predicts customer churn probability based on various customer metrics
- Interactive web interface
- Real-time predictions
- Visual representation of churn probability
- Responsive design

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Run the Flask application:
```bash
python app.py
```

3. Open your browser and navigate to `http://localhost:5000`

## Input Features

- Tenure
- Warehouse to Home distance
- Number of Devices Registered
- Preferred Order Category
- Satisfaction Score
- Number of Addresses
- Complain History
- Order Amount Hike
- Coupon Usage
- Order Count
- Days Since Last Order

## Technologies Used

- Flask
- Scikit-learn
- Pandas
- Bootstrap
- JavaScript


# if you want how to load and view a .jpblib files in python
import joblib
# Load the joblib file
model = joblib.load('path/to/your/file.joblib')
# Print the model or inspect its attributes
print(model)

# if you want how to load and view a .csv files in python
import pandas as pd
# Load the CSV file
data = pd.read_csv('path/to/your/file.csv')
# Print the first few rows of the dataframe
print(data.head())

git rm --cached *.csv *.xlsx *.joblib
git commit -m "Remove ignored files from repository"
git push origin master