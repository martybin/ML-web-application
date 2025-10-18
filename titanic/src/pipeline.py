from utils import *
from preprocessing import Preprocessing

class ModelTraining:
    def __init__(self):
        self.preprocessor = Preprocessing()
        self.X = self.preprocessor.X
        self.y = self.preprocessor.y

    def pipeline(self):
        numerical_pipeline = Pipeline([('scaler', StandardScaler())])
        categorical_pipeline = Pipeline([('OneHot', OneHotEncoder(handle_unknown='ignore'))])
        transformer = ColumnTransformer([
            ('num', numerical_pipeline, ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'CabinNumber']),
            ('cat', categorical_pipeline, ['Sex', 'Embarked', 'CabinClass'])
        ])
        return transformer

    def split_data(self):
        return train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def train_model(self):
        X_train, X_test, y_train, y_test = self.split_data()
        return X_train, X_test, y_train, y_test

    def mlpipeline(self):
        pipe = self.pipeline()
        X_train, X_test, y_train, y_test = self.train_model()
        mlpipe = Pipeline([
            ('initial_preprocess', self.preprocessor),
            ('transformer', pipe),
            ('xgb', XGBClassifier()),
        ])
        mlpipe.fit(X_train, y_train)
        return mlpipe

    def predict_model(self):
        X_train, X_test, y_train, y_test = self.train_model()
        mlpipe = self.mlpipeline()
        Y_hat = mlpipe.predict(X_test)
        return Y_hat, y_test

    def p_score(self):
        yhat, y_test = self.predict_model()
        precision = precision_score(y_test, yhat)
        return f'The precision score is: {precision:.3f}'

    def save_pipeline(self, file_path='xgbpipe.joblib'):
        mlpipe = self.mlpipeline()
        joblib.dump(mlpipe, file_path)
        return f'Pipeline saved as {file_path}'
