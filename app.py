from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
import joblib
import os

app = Flask(__name__)

def prepare_model():
    # Check if CSV file exists
    if not os.path.exists('E Commerce Dataset.csv'):
        print("Warning: E Commerce Dataset.csv not found. Creating dummy model for deployment.")
        # Create dummy model components for deployment
        categorical_features = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender',
                              'PreferedOrderCat', 'MaritalStatus']
        numerical_features = ['Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp',
                             'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
                             'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed',
                             'OrderCount', 'DaySinceLastOrder', 'CashbackAmount']
        features = categorical_features + numerical_features

        # Create dummy label encoders
        label_encoders = {}
        for feature in categorical_features:
            le = LabelEncoder()
            # Fit with dummy data
            if feature == 'PreferredLoginDevice':
                le.fit(['Mobile Phone', 'Computer', 'Phone'])
            elif feature == 'PreferredPaymentMode':
                le.fit(['Debit Card', 'Credit Card', 'UPI', 'Cash on Delivery', 'E wallet'])
            elif feature == 'Gender':
                le.fit(['Male', 'Female'])
            elif feature == 'PreferedOrderCat':
                le.fit(['Laptop & Accessory', 'Mobile Phone', 'Fashion', 'Grocery', 'Others'])
            elif feature == 'MaritalStatus':
                le.fit(['Single', 'Married', 'Divorced'])
            label_encoders[feature] = le

        # Create dummy model, scaler, and imputer
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        # Create dummy training data
        import numpy as np
        X_dummy = np.random.rand(100, len(features))
        y_dummy = np.random.randint(0, 2, 100)

        imputer = KNNImputer(n_neighbors=5)
        X_dummy_imputed = imputer.fit_transform(X_dummy)

        scaler = StandardScaler()
        X_dummy_scaled = scaler.fit_transform(X_dummy_imputed)

        model.fit(X_dummy_scaled, y_dummy)

        # Save all objects
        joblib.dump(model, 'model.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        joblib.dump(imputer, 'imputer.joblib')
        joblib.dump(label_encoders, 'label_encoders.joblib')
        joblib.dump(features, 'features.joblib')
        return

    # Original model training code when CSV exists
    df = pd.read_csv('E Commerce Dataset.csv')

    # Create label encoders for categorical variables
    categorical_features = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender',
                          'PreferedOrderCat', 'MaritalStatus']

    label_encoders = {}
    for feature in categorical_features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature].astype(str))
        label_encoders[feature] = le

    # Prepare numerical features
    numerical_features = ['Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp',
                         'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
                         'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed',
                         'OrderCount', 'DaySinceLastOrder', 'CashbackAmount']

    # Combine features
    features = categorical_features + numerical_features

    # Prepare X and y
    X = df[features]
    y = df['Churn']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize imputer and scaler
    imputer = KNNImputer(n_neighbors=5)
    X_train_imputed = imputer.fit_transform(X_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,  # Reduced from 200 for faster training
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Save all objects
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(imputer, 'imputer.joblib')
    joblib.dump(label_encoders, 'label_encoders.joblib')
    joblib.dump(features, 'features.joblib')
    
# Load or prepare model
try:
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    imputer = joblib.load('imputer.joblib')
    label_encoders = joblib.load('label_encoders.joblib')
    features = joblib.load('features.joblib')
    print("Successfully loaded existing model files.")
except Exception as e:
    print(f"Model files not found ({e}). Preparing new model...")
    prepare_model()
    try:
        model = joblib.load('model.joblib')
        scaler = joblib.load('scaler.joblib')
        imputer = joblib.load('imputer.joblib')
        label_encoders = joblib.load('label_encoders.joblib')
        features = joblib.load('features.joblib')
        print("Successfully created and loaded new model.")
    except Exception as e2:
        print(f"Failed to create model: {e2}")
        # Initialize with None values as fallback
        model = None
        scaler = None
        imputer = None
        label_encoders = None
        features = None

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                             'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model is loaded
        if model is None or scaler is None or imputer is None or label_encoders is None or features is None:
            return jsonify({
                'success': False,
                'error': 'Model not properly initialized. Please check server logs.'
            })

        data = request.json

        # Create input data dictionary
        input_data = {}

        # Process categorical features
        categorical_features = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender',
                              'PreferedOrderCat', 'MaritalStatus']
        for feature in categorical_features:
            if feature in data:
                try:
                    le = label_encoders[feature]
                    input_data[feature] = le.transform([str(data[feature])])[0]
                except ValueError:
                    # Handle unknown categorical values
                    input_data[feature] = 0
            else:
                input_data[feature] = 0  # Default value for missing categorical data

        # Process numerical features
        numerical_features = ['Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp',
                            'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
                            'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed',
                            'OrderCount', 'DaySinceLastOrder', 'CashbackAmount']

        for feature in numerical_features:
            input_data[feature] = float(data.get(feature, 0))  # Default value 0 for missing numerical data

        # Create DataFrame with proper column order
        input_df = pd.DataFrame([input_data])[features]

        # Handle missing values and scale
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)

        # Make prediction
        prediction = model.predict_proba(input_scaled)
        churn_probability = round(float(prediction[0][1]) * 100, 2)

        return jsonify({
            'success': True,
            'churn_probability': churn_probability
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
