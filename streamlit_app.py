import streamlit as st
import pandas as pd
st.title('🤖machine learning')

st.info("this is a machine learning application for test")
df=pd.read_csv("penguins_cleaned.csv")
df

