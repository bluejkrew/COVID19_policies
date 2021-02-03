import streamlit as st
import pandas as pd 
import altair as alt 


def app():
	st.title("Targeted Strategies in Reducing the Spread of COVID-19 and Future Pandemics")
	# And write the rest of the app inside this function!
	st.sidebar.header("Header 1")
	st.markdown('''This is where I place my paragraph.''')

	st.code("x = 8 + 13")

	st.sidebar.markdown('''Ahh yeah...bullet points!''')
if __name__ == '__main__':
	app()