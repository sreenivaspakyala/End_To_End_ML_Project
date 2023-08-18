import os, sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import  load_object

# allowed values as inputs
# The unique values in column gender are: ['female' 'male']
# The unique values in column race/ethnicity are: ['group B' 'group C' 'group A' 'group D' 'group E']
# The unique values in column parental level of education are: ["bachelor's degree" 'some college' "master's degree" "associate's degree"
#  'high school' 'some high school']
# The unique values in column lunch are: ['standard' 'free/reduced']
# The unique values in column test preparation course are: ['none' 'completed']
# numeric column values range between 0 - 100 ['reading score', 'writing score']

class PredictPipeline():
    def __init__(self) -> None:
        self.model_path = os.path.join(os.getcwd(),'artifacts','model.pkl')
        self.preprocessor_path = os.path.join(os.getcwd(),'artifacts','preprocessor.pkl')
        logging.info('Predict Pipeline Initiated.')
    
    def predict_data(self, features):
        try:
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)
            transformed_features = preprocessor.transform(features)
            prediction = model.predict(transformed_features)

            return prediction
        
        except Exception as err:
            logging.error('An Error has occurred during Prediction.')
            raise CustomException(err, sys)



class CustomData():
    def __init__(self, gender:str, race_ethnicity:str, parental_level_of_education:str, 
                 lunch:str, test_preparation_course:str, reading_score:int, writing_score:int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    
    def convert_to_df(self):

        try:
            # while creating df for a single row make sure you pass values as a list not scaler values
            # if you pass single value if throws a error sying pass an index
            data_dict = {
                'gender' : [self.gender],
                'race/ethnicity' : [self.race_ethnicity],
                'parental level of education' : [self.parental_level_of_education],
                'lunch' : [self.lunch],
                'test preparation course' : [self.test_preparation_course],
                'reading score' : [self.reading_score],
                'writing score' : [self.writing_score]
            }

            return pd.DataFrame(data_dict)
        
        except Exception as err:
            raise CustomException(err, sys)

# for testing purpose before making a front end simulation.

if __name__ == '__main__':
    MENU = '''
Please follow the below Instruction while providing Inputs:
Allowed values as inputs
The unique values in column gender are: ['female' 'male']
The unique values in column race/ethnicity are: ['group B' 'group C' 'group A' 'group D' 'group E']
The unique values in column parental level of education are: ["bachelor's degree" 'some college' "master's degree" "associate's degree" 'high school' 'some high school']
The unique values in column lunch are: ['standard' 'free/reduced']
The unique values in column test preparation course are: ['none' 'completed']
numeric column values range between 0 - 100 ['reading score', 'writing score']
    
'''
    inputs_list = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course',
                'reading score',
                'writing score' 
    ]
    inputs = []
    print(MENU)

    for i in range(len(inputs_list)):
        value = input(f'Enter the input for {inputs_list[i]}: ')
        inputs.append(value)
    

    data = CustomData(*inputs)
    df = data.convert_to_df()
    print(df)
    result = round(PredictPipeline().predict_data(df)[0],2)
    print(result)
    print(f'The Math Score is: {result} marks.')  # "{:.2f}.format(number) -> formats the number to 2 places."



