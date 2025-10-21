from dependencies import *

class Preprocessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ageImputer = SimpleImputer(strategy='mean')

    def fit(self, X, y=None):
        self.ageImputer.fit(X[['Age']])
        return self

    def transform(self, X, y=None):
        df = X.copy()
        df['Age'] = self.ageImputer.transform(df[['Age']])
        df['CabinClass'] = df['Cabin'].fillna('M').apply(lambda x: str(x).replace(' ', '')).apply(lambda x: re.sub(r'[^a-zA-Z]', '', str(x)))
        df['CabinNumber'] = df['Cabin'].fillna('M').apply(lambda x: str(x).replace(' ', '')).apply(lambda x: re.sub(r'[^0-9]', '', str(x))).replace('', 0).astype(int)
        df['Embarked'] = df['Embarked'].fillna('M')
        df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
        return df
