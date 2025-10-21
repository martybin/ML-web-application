from dependencies import *
from preprocessing import Preprocessing

class ModelTraining: 
    def __init__(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'data', 'titanic.csv')
        df = pd.read_csv(path)
        self.X = df.drop('Survived', axis=1)
        self.y = df['Survived']
        self.mlpipe = None

    def pipeline(self):
        numerical_pipeline = Pipeline([('scaler', StandardScaler())])
        categorical_pipeline = Pipeline([('OneHot', OneHotEncoder(handle_unknown='ignore'))])
        transformer = ColumnTransformer([
            ('num', numerical_pipeline, ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'CabinNumber']),
            ('cat', categorical_pipeline, ['Sex', 'Embarked', 'CabinClass'])
        ])
        return transformer

    def mlpipeline(self):
        preprocessor = Preprocessing()
        transformer = self.pipeline()
        mlpipe = Pipeline([
            ('initial_preprocess', preprocessor),
            ('transformer', transformer),
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
        ])
        return mlpipe

    def train_test_eval(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.mlpipe = self.mlpipeline()
        self.mlpipe.fit(X_train, y_train)
        yhat = self.mlpipe.predict(X_test)
        precision = precision_score(y_test, yhat)
        print(f'Precision: {precision:.3f}')
        return self.mlpipe

    def save_pipeline(self, file_path='xgbpipe.joblib'):
        if self.mlpipe is None:
            print("Pipeline not trained yet. Training now...")
            self.train_test_eval()

        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', file_path))
        joblib.dump(self.mlpipe, file_path)
        print(f'Pipeline trained & saved at: {file_path}')
        return file_path
