import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import streamlit as st
from utils import PreProcessor, column
from src.load_data import load_pipeline

model = load_pipeline()
st.title('Will you survive if you were among Titanic passengers or not? :ship:')

passengerid = st.text_input('Input Passenge ID', '8585')
pclass = st.selectbox('Choose Class', [1, 2, 3])
name = st.text_input('Input Passenger Name')
sex = st.select_slider('Choose Sex', ['Male', 'Female'])
age = st.slider('Choose Age', 0, 100)
sibsp = st.slider('Choose Sibilings', 0, 10)
parch = st.slider('Choose count of parents & children', 0, 10)
ticket = st.text_input('Input Ticket Number', '8585')
fare = st.number_input('Input Fare Price', 0, 1000)
cabin = st.text_input('Input Cabin', 'C52')
embarked = st.selectbox('Did They Embark?', ['S', 'C', 'Q'])

def predict():
    row = np.array([passengerid, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked])
    X = pd.DataFrame([row], columns=column)
    prediction = model.predict(X)
    if prediction[0] == 1:
        st.success('Passenger Survived :thumbsub:')
    else:
        st.error('Passenger did not Survive :thumbsdown:')

trigger = st.button('Predict', on_click=predict)
