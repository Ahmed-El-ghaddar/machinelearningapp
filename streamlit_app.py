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

