import os
from pipeline import ModelTraining

if __name__ == '__main__':
    file_path = 'xgbpipe.joblib'
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', file_path))
    trainer = ModelTraining()
    trainer.save_pipeline(file_path=save_path)
