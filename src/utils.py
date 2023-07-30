import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill

from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(X_train,y_train,X_test,y_test,models,score_fn,params):
    try:
        report = {}

        for i in range(len(models)):

            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            grid_cv = GridSearchCV(estimator=model, 
                                         param_grid=param,
                                         cv = 5, verbose = 2, 
                                         n_jobs=1)
            grid_cv.fit(X_train, y_train)

            model.set_params(**grid_cv.best_params_)

            model.fit(X_train, y_train)

            y_pred_tr = model.predict(X_train)

            y_pred_ts = model.predict(X_test)

            train_score = score_fn(y_train, y_pred_tr)
            test_score = score_fn(y_test, y_pred_ts)

            report[list(models.keys())[i]] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    

