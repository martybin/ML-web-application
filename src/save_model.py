import os
from src.pipeline import ModelTraining

if __name__ == '__main__':
    file_path = 'xgbpipe.joblib'
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', file_path))
    trainer = ModelTraining()
    trainer.save_pipeline(file_path=file_path)
