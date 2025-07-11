#!/bin/bash
echo "Starting Customer Churn Prediction App..."
echo "Python version:"
python --version
echo "Current directory:"
pwd
echo "Files in directory:"
ls -la
echo "Starting Flask app..."
exec python start.py
