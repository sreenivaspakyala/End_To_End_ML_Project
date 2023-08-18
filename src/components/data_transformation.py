import os, sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocesser_file_path:str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformer:
    def __init__(self) -> None:
        self.datatransformationconfig = DataTransformationConfig()
    

    def get_data_transformer(self):
        try:
            numerical_columns = ['reading score', 'writing score']
            categorical_columns = [
                'gender',
                'race/ethnicity', 
                'parental level of education', 
                'lunch', 
                'test preparation course'
            ]

            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler(with_mean=False))
                    ]
            )
            logging.info('Numerical Pipeline created.')

            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoding',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            logging.info('Categorical Pipeline created.')

            preprocessor = ColumnTransformer(
                [
                    ('numerical_pipeline',numerical_pipeline,numerical_columns),
                    ('categorical_pipeline',categorical_pipeline,categorical_columns)

                ]
            )

            return preprocessor


        except Exception as err:
            logging.error('An Error has occurred during Data transformer creation.')
            raise CustomException(err, sys)
    

    def start_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info('Train and Test data frames are created.')
            logging.info('Fetching the preprocessor object.')

            preprocessor_obj = self.get_data_transformer()

            target_column = 'math score'
            numerical_columns = ['reading score', 'writing score']

            input_train_df = train_df.drop(columns=[target_column],axis=1)
            target_train_df = train_df[target_column]

            input_test_df = test_df.drop(columns=[target_column],axis=1)
            target_test_df = test_df[target_column]

            logging.info('Applying preprocessor obj on train & test data.')

            input_train_arr = preprocessor_obj.fit_transform(input_train_df)
            input_test_arr = preprocessor_obj.transform(input_test_df)

            train_arr = np.c_[input_train_arr,np.array(target_train_df)]
            test_arr = np.c_[input_test_arr,np.array(target_test_df)]

            # saving the preprocessor pickle file
            logging.info('Saving the preprocessor Object.')

            save_object(self.datatransformationconfig.preprocesser_file_path, preprocessor_obj)

            return (train_arr,
                    test_arr,
                    self.datatransformationconfig.preprocesser_file_path
                    )
        
        except Exception as err:
            logging.error('An Error has occures during Data Transformation.')
            raise CustomException(err, sys)