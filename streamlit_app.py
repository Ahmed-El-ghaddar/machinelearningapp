import streamlit as st
import pandas as pd
st.title('🤖machine learning')
st.info("this is a machine learning application for test")
with st.expander("Data"):
  st.write('** Row Data')
  df=pd.read_csv("penguins_cleaned.csv")
  df
  st.write('***x***')
  x=df.drop('species',axis=1)
  x
  st.write('***Y***')
  y=df.species
  y
with st.expander('Data visualuzation'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

with st.sidebar:
  st.header("Input features")
  st.header('Input features')
  island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
  bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
  bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
  flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
  body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
  gender = st.selectbox('Gender', ('male', 'female'))
