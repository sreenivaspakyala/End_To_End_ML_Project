import os, sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass   # to make class creation easy 

from src.components.data_transformation import DataTransformationConfig, DataTransformer
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer



@dataclass  # such data class is useful mainly when you only variables as part of a class
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','rawdata.csv')

class DataIngestion:
    def __init__(self) -> None:
        self.data_ingestion_config = DataIngestionConfig()
    
    def start_data_ingestion(self):
        logging.info('Started Data Ingestion process...')
        try:
            df = pd.read_csv(r'.\notebook\data\StudentsPerformance.csv')
            logging.info('Reading dataset into a dataframe.')
            # create folder structure for train & test data

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            # save the dataset as raw data
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True)
            logging.info('Rawdata has been saved.')
            logging.info('Initiating Train Test split')

            train_data, test_data = train_test_split(df,test_size=0.25,random_state=75)
            logging.info('Saving Train & Test data.')

            train_data.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)
            logging.info('Data Ingestion is completed.')

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as err:
            logging.error('An Error has occurred during Data Ingestion.')
            raise CustomException(err, sys)


if __name__=='__main__':
    test_obj = DataIngestion()
    train_path, test_path = test_obj.start_data_ingestion()
    data_transformation = DataTransformer()
    train_arr, test_arr, _ = data_transformation.start_data_transformation(train_path, test_path)
    model_trainer = ModelTrainer()
    print(model_trainer.start_model_trainer(train_arr, test_arr))
