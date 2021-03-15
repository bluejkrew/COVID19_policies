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
	
	# Pandas example
  	st.markdown('''## Pandas Streamlit is made to render lots of Python objects such as Pandas DataFrames. Let's create a simple DataFrame and render it.''')
  	df = pd.DataFrame({'x':[1,2,3], 'y':[40, 20, 30], 'z':[100, 40, 33]})
  	st.write(df)
  	st.markdown("""Remember that Streamlit will run any Python code you give it, so you can of course manipulate your DataFrame. Also, you can use Streamlit widgets for simple control flow and interactivity! Let's put some of these in the sidebar. We'll use the following code:
  	
  	```python
  	func = st.sidebar.selectbox("Aggregate:", ['Sum', 'Mean', 'Median'])
	
	
	
if __name__ == '__main__':
	app()
