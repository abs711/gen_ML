import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Train Test split")

            X_train, y_train,X_test,y_test=(train_array[:,:-1],
                                            train_array[:,-1],
                                            test_array[:,:-1],
                                            test_array[:,-1],)
            models = {
                "Random Forest":RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBoost Regressor":  XGBRegressor(),
                "CatBoost Regressor":CatBoostRegressor(verbose=False),
                "AdaBoost Regressor" :AdaBoostRegressor()
            }

            params = {
                "Random Forest":{'n_estimators': [8,16,32,64,128,256],
                                 'criterion': ['squared_error',
                                               'friedman_mse',
                                               'absolute_error']                                               
                                },

                "Decision Tree":{'max_features': ['sqrt','log2']
                                },

                "Gradient Boosting":{'learning_rate': [.1,.01,.001],
                                 'subsample': [0.6,0.7,0.75,0.8,0.85,0.9],
                                 'n_estimators': [8,16,32,64,128,256]
                                },
                "Linear Regression":{},
                "K-Neighbors Regressor": {},
                "XGBoost Regressor":  {'learning_rate': [.1,.01,.001],
                                 'n_estimators': [8,16,32,64,128,256]},
                "CatBoost Regressor":{'learning_rate': [.1,.01,.001],
                                 'depth': [6,8,10],
                                 'iterations': [30, 50, 100]},
                "AdaBoost Regressor" :{'learning_rate': [.1,.01,.001],
                                 'n_estimators': [8,16,32,64,128,256]}
            }


            score_fn = r2_score

            model_report:dict=evaluate_model(X_train=X_train,
                                             y_train=y_train,
                                             X_test=X_test,
                                             y_test=y_test,
                                             models=models,score_fn=score_fn,params=params)
            

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("All Bad Models")
            
            logging.info("Register best model info")

            save_object(file_path=self.model_trainer_config.trained_model_path,
                        obj=best_model
                        )
            
            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)