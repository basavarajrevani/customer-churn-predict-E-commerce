#!/usr/bin/env python3
"""
Startup script for the Customer Churn Prediction app
This ensures the app starts correctly on Render
"""
import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the Flask app
if __name__ == "__main__":
    from app import app
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Customer Churn Prediction app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
