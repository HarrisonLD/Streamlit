
import pickle
import datetime
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import OrdinalEncoder

html_temp = """
<div style="background-color:tomato;padding:1.5px">
<h1 style="color:white;text-align:center;">Single Customer </h1>
</div><br>"""
st.sidebar.markdown(html_temp, unsafe_allow_html=True)

html_temp = """
<div style="background-color:tomato;padding:1.5px">
<h1 style="color:white;text-align:center;">Single Customer </h1>
</div><br>"""
st.markdown(html_temp, unsafe_allow_html=True)


filename = 'Auto_Price_Pred_Model'
# model = pickle.load(open(filename, 'rb'))

with st.sidebar:
    st.subheader('Car Specs to Predict Price')

# sidebars for customer inputs
age = st.sidebar.number_input("What is the age of your car:", 0,1,2,3) 
hp_kw = st.sidebar.slider("What is the Horse Power of your car:", 60, 360, step=6)
km = st.sidebar.slider(" What is the km for your car:",0, 350000, step=5000)
Gearing_Type = st.sidebar.radio("Gearing Type", ("Manual", "Automatic", "Semi-automatic"))
car_model = st.sidebar.selectbox("Model Selection", ("Audi A3", "Audi A1", "Opel Insignia",
                                 "Opel Astra", "Opel Corsa", "Renault Clio", "Renault Espace", "Renault Duster"))


lavonda_transformer = pickle.load(open('transformer', 'rb'))
lavonda_model = pickle.load(open('rf_model_new', 'rb'))

my_dict = {
    "make_model": "Car Model",
    "hp_kW": "Horse Power",
    "age": "Age",
    "km": "km Traveled",
    "Gearing_Type": "Gearing Type"
}

df = pd.DataFrame.from_dict([my_dict])
st.table(df)

st.header("Car Configuration is Below")

df2 = lavonda_transformer.transfrom(df)

st.subheader('This is a Streamlit app for Car Configuration.')

if st.button("Predict"):
    prediction = lavonda_model.predict(df2)
    st.success(" The estimated proice of your car is []. ".format(int(predict[0])))
