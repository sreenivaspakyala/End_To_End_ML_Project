import pickle as pk
import os, sys
from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pk.dump(obj, file_obj)
        
    except Exception as err:
        logging.error('An Error has occurred while saving Object.')
        raise CustomException(err, sys)

def evaluate_models(x_train, y_train, x_test, y_test, models:dict, params:dict):
    result = {}
    try:
        for i in range(len(models.keys())):
            model = list(models.values())[i]
            parameter = params[list(models.keys())[i]]

            grcv = GridSearchCV(model, parameter, n_jobs=3, cv=3, verbose=False)
            grcv.fit(x_train, y_train)

            model.set_params(**grcv.best_params_)
            model.fit(x_train, y_train)

            # will implement hyperparameter tuning part here
            # using gridsearch CV

            train_pred = model.predict(x_train)
            test_pred = model.predict(x_test)

            train_score = r2_score(y_train,train_pred)
            test_score = r2_score(y_test,test_pred)

            result[list(models.keys())[i]] = test_score

        return result

    except Exception as err:
        logging.error('An Error has occurred during Model Evaluation.')
        raise CustomException(err, sys)
