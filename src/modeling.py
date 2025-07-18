from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


class ModelFactory:
    def get_models(self):
        return {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'ElasticNet': ElasticNet(),
            'RandomForest': RandomForestRegressor(random_state=42),
            'ExtraTrees': ExtraTreesRegressor(random_state=42),
            'GradientBoosting': GradientBoostingRegressor(random_state=42),
            'AdaBoost': AdaBoostRegressor(random_state=42),
            'SVR': SVR(),
            'DecisionTree': DecisionTreeRegressor(random_state=42),
            'XGBoost': XGBRegressor(random_state=42)
        }

class ModelTrainer:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model
