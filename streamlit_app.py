#
#
#
import streamlit as st
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open('catboost_model.pkl','rb'))

st.title("Insurance Premium Prediction")

## for Gender Column
sex = st.selectbox('Please select gender', ('male', 'female'))

## for age Column
#age = st.text_input('Enter Age', 18)
age = st.number_input('Enter Age', min_value=0, max_value=90, value=18, step=1)
age = int(age)



# for BMI Column
bmi = st.number_input('Enter BMI', min_value=10.0, max_value=50.0, value=18.0, step=0.1)
bmi = float(bmi)

## for children Column
children = st.number_input('Enter no of childrens', min_value=0, max_value=5, value=2, step=1)
children = int(children)

## for smokers Column
smoker = st.selectbox('Please select smoker category ', ("yes","no"))

## for region Column
region = st.selectbox('Please select region ', ("southwest", "southeast", "northwest", "northeast"))

l = {}
l['age'] = age
l['sex'] = sex
l['bmi'] = bmi
l['children'] = children



l['smoker'] = smoker
l['region'] = region

df = pd.DataFrame(l, index=[0])

df['region'] = df['region'].map({'southwest':3, 'southeast':2, 'northwest':1, 'northeast':0})
df['sex'] = df['sex'].map({'male':1, 'female':0})
df['smoker'] = df['smoker'].map({'yes':1, 'no':0})

y_pred = model.predict(df)

if st.button("Show Result"):
    st.header(f" Insurance Prediction is  {round(y_pred[0],2)} INR")
