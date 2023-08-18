import os, sys
import streamlit as st

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# let's create a user input page for the model prediction.

st.title('Welcome to Student Performance Predictor :book: :100:')
st.text('Predict the Student performance by filling below inputs...')

with st.form('Student_Details'):
    st.write('Please fill the Details below :point_down:')

    gender = st.radio('Gender  	:male-student: 	:female-student:', options=['male', 'female'])
    race = st.radio('Race/Ethnicity', ['group A', 'group B', 'group C', 'group D', 'group E'])
    education = st.selectbox('Parental Level of Education', 
                 ['bachelor\'s degree', 'some college', 'master\'s degree', 
                  'associate\'s degree','high school', 'some high school'])
    lunch = st.radio('Lunch :sandwich:', ['standard', 'free/reduced'])
    course = st.checkbox('Completed Test Preparation Course :book:')
    reading_score = st.slider('Reading Score 	:100:', min_value=0, max_value=100)
    writing_score = st.slider('Writing Score 	:100:', min_value=0, max_value=100)

    if st.form_submit_button('Generate Score'):
        if course:
            course = 'completed'
        else:
            course = 'none'
        inputs = [gender, race, education, lunch, course, reading_score, writing_score]
        df = CustomData(*inputs).convert_to_df()
        result = round(PredictPipeline().predict_data(df)[0],2)

        st.write('The Predicted Math Score for the given student details is ', result, ' marks')

st.write('	:heart:  Developed by Sreenivas Pakyala  :heart:')
st.write(':star: :star: Thanks for using the App :star: :star:')        