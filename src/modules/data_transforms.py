import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer #transform pipeline
from sklearn.impute import SimpleImputer #handling missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact',"preprocessor.pkl")

class DataTransformation:    
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            #numerical features tranforms
            numerical_pipeline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            
            logging.info("Numerical columns scaling done")

            #categorical features transforms
            categorical_pipeline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]

            )
            
            logging.info("Catgorical columns encoding done")


            preprocessor=ColumnTransformer(
                [
                    ("numerical_pipeline",numerical_pipeline,numerical_columns),
                    ("categorical_pipeline",categorical_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self, train_path,test_path):
        
        try:
            logging.info("Reading train and test data")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Getting preprocessor")

            preprocessor = self.get_data_transformer()
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feat_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feat_train_df = train_df[target_column_name]

            input_feat_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feat_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing to train and test dfs")

            input_feat_train_arr=preprocessor.fit_transform(input_feat_train_df)
            input_feat_test_arr=preprocessor.transform(input_feat_test_df)

            train_arr = np.c_[input_feat_train_arr, np.array(target_feat_train_df)]
            test_arr = np.c_[input_feat_test_arr, np.array(target_feat_test_df)]

            logging.info(f"Saved preprocessor")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,

            )
        except Exception as e:
            raise CustomException(e, sys)
