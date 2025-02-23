import os

files_to_remove = ['model.joblib', 'scaler.joblib', 'imputer.joblib', 'label_encoders.joblib', 'features.joblib']
for file in files_to_remove:
    try:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed {file}")
    except Exception as e:
        print(f"Error removing {file}: {e}")
