import streamlit as st
import pandas as pd 
import altair as alt 


def app():
	st.title("Targeted Strategies in Reducing the Spread of COVID-19 and Future Pandemics")
	# And write the rest of the app inside this function!
	st.sidebar.header("Header 1")
	st.markdown('''This website is intended to assist public health officials and decision makers in identifying effective policies to combat a respiratory-spread pandemic based on county-specific features.

The model is based on county-level disease data collected during the COVID-19 pandemic, combined with census data for county-level demographics. 
.''')

	st.code("x = 8 + 13")

	st.sidebar.markdown('''Ahh yeah...bullet points!''')
if __name__ == '__main__':
	app()