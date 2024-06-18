import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle

# Load trained file
model = pickle.load(open('BO_Revenue_Predictive_Model.pkl', 'rb'))

# Title of app
st.title('Movie Worldwide Revenue Prediction')

# Description
st.write('This app predicts the worldwide gross revenue of movies based on inputs like production budget, movie rating, and genre.')

# Collect user input
constant = float(1)
budget = st.number_input('Enter the production budget ($):', min_value=0, max_value=10000000000, value=10000000)
num_votes = st.slider('Enter rating (0.0 to 10.0):', min_value=0.00, max_value=10.00, value=5.00, step=0.01, format="%.2f")
genres = ['Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Musical', 'Romance', 'Sci-Fi']
selected_genre = st.selectbox('Select the genre:', genres)
num_votes_transformed = (num_votes/10)*1000000


# Dummy encoding for genres
genre_data = {genre: 0 for genre in genres} # Initialize all genres to 0
genre_data[selected_genre] = 1 # Set selected genre to 1
 
# Prepare featurs for prediction
features = [constant, budget, num_votes_transformed] + list(genre_data.values())
features = np.array(features)


# Predict button
if st.button('Predict Revenue'):
    prediction = model.predict(features.reshape(1, -1))
    st.success(f'Estimated Worldwide Gross Revenue: ${prediction[0]:,.2f}')



st.markdown('## Model Information')
st.text('This model is build using a least squares regression method to estimate ')
st.text('potential movie revenues based on historical data')

