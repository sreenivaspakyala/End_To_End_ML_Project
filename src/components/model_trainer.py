import numpy as np
import pandas as pd
import os, sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    model_file_path:str = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()
    
    def start_model_trainer(self, train_data, test_data):
        logging.info('Splitting Training & Testing data.')

        try:
            x_train, y_train, x_test, y_test = (
                train_data[:,:-1],
                train_data[:,-1],
                test_data[:,:-1],
                test_data[:,-1]
            )

            all_models = {
                'Linear Regression': LinearRegression(),
                'KNeighbour regression': KNeighborsRegressor(),
                'Catboost': CatBoostRegressor(verbose=0), # verbose to avoid extra output.
                'XGboost': XGBRegressor(),
                'Decision tree Regression': DecisionTreeRegressor()
            }
            
            # implement params here for Hyperparameter tuning
            params = {
                'Linear Regression': {},
                'KNeighbour regression': {
                    'n_neighbors': [5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                'Catboost': {
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                'XGboost': {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'Decision tree Regression': {
                    "splitter":["best","random"],
                    "max_depth" : [1,3,5,7,9,11,12],
                    "max_features":["log2","sqrt"],
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                }
            }


            # evaluates all the above mentioned models
            models_result = evaluate_models(x_train, y_train, x_test, y_test, all_models, params)

            # highest r2 score among the models
            best_score = max(sorted(models_result.values()))

            # model having the best r2 score
            best_model_name = list(all_models.keys())[
                list(models_result.values()).index(best_score)
            ]

            # to set a threshold for the r2 score for all models
            if best_score < 0.6:
                raise CustomException('No best Model has been found', sys)
            
            logging.info('Best Model has been found.')
            best_model = all_models[best_model_name]

            logging.info('Saving the best model.')

            save_object(self.model_trainer_config.model_file_path, best_model)

            logging.info('Best Model has been saved Successfully.')

            predictions = best_model.predict(x_test)
            pred_score = r2_score(y_test, predictions)

            return (pred_score, best_model_name)

        except Exception as err:
            logging.error('An Error has occurred during Best Model Evaluation.')
            raise CustomException(err, sys)

