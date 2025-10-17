from src.utils import *

class Preprocessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.df = sns.load_dataset('titanic')
        self.X = self.df.drop('Survived', axis=1)
        self.y = self.df.Survived

    def fit(self, X, y=None):
        self.ageImputer = SimpleImputer()
        self.ageImputer.fit(self.X[['Age']])
        return self

    def transform(self, X, y=None):

        self.X['Age'] = self.ageImputer.transform(self.X[['Age']])
        self.X['CabinClass'] = self.X['Cabin'].fillna('M').str.apply(lambda x: str(x).replace(' ', '')).apply(lambda x: re.sub(r'[^a-zA-Z]', '', x, regex=True))
        self.X['CabinNumber'] = self.X['Cabin'].fillna('M').str.apply(lambda x: str(x).replace(' ', '')).apply(lambda x: re.sub(r'[^0-9]', '', x, regex=True)).replace('', 0)
        self.X['Embarked'] = self.X['Embarked'].fillna('M')
        self.X = self.X.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
        return self.X
