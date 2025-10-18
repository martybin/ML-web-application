from utils import *

class Preprocessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.path = os.path.join(os.path.dirname(__file__), '..', 'data', 'titanic.csv')
        self.df = pd.read_csv(self.path)

    def fit(self, X, y=None):
        self.ageImputer = SimpleImputer()
        self.ageImputer.fit(self.X[['Age']])
        return self

    def transform(self, X, y=None):
        df = self.df.copy()
        df['Age'] = self.ageImputer.transform(df[['Age']])
        df['CabinClass'] = df['Cabin'].fillna('M').apply(lambda x: str(x).replace(' ', '')).apply(lambda x: re.sub(r'[^a-zA-Z]', '', x))
        df['CabinNumber'] = df['Cabin'].fillna('M').apply(lambda x: str(x).replace(' ', '')).apply(lambda x: re.sub(r'[^0-9]', '', x)).replace('', 0)
        df['Embarked'] = df['Embarked'].fillna('M')
        df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
        X = df.drop('Survived', axis=1)
        y = df['Survived']
        return X, y
