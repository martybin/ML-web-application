import os
import joblib

def load_pipeline(file_path='xgbpipe.joblib'):
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', file_path))
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Model file not found at {file_path}')
    
    mlpipe = joblib.load(file_path)
    print(f'Pipeline loaded successfully from: {file_path}')
    return mlpipe
