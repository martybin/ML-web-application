import joblib

def load_pipeline(file_path='xgbpipe.joblib'):
    mlpipe = joblib.load(file_path)
    print(f"Pipeline loaded from {file_path}")
    return mlpipe
