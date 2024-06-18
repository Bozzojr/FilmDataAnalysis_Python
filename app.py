import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle

# Load trained model
model = pickle.load(open('BO_Revenue_Predictive_Model.pkl', 'rb'))

# Define app title and description
st.set_page_config(page_title="Movie Revenue Predictor", page_icon=":clapper:", layout="wide")

# CSS for central alignment and background image
st.markdown(
    """
    <style>
    .center-align {
        text-align: center;
    }
    .banner-image {
        width: 100%;
        height: 300px;
        margin-bottom: 20px;
    }
    .stApp {
        background: url('https://img.freepik.com/free-photo/popcorn-line-near-box-glasses_23-2147698910.jpg?size=626&ext=jpg&ga=GA1.2.1710465407.1718743044&semt=ais_user');
        background-size: cover;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        background: rgba(255, 255, 255, 0.8);  /* Adding transparency to the content background */
        border-radius: 10px;
    }
    .input-container {
        max-width: 600px;
        margin: auto;
    }
    </style>
    """, unsafe_allow_html=True
)



# Display title
st.markdown('<h1 class="center-align">🎥 Movie Worldwide Revenue Prediction</h1>', unsafe_allow_html=True)

# Display description
st.markdown(
    """
    <div class="center-align">
    Welcome to the Movie Worldwide Revenue Predictor! This application uses a least squares regression model to estimate the potential worldwide gross revenue of movies based on key input features such as production budget, movie rating, and genre.
    </div>
    """, unsafe_allow_html=True
)

# Use a single container for inputs and center it
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)

    # Collect user input for budget
    budget = st.number_input('💰 Enter the production budget ($):', min_value=0, max_value=1000000000, value=10000000, step=1000000)
    
    # Collect user input for rating
    num_votes = st.slider('⭐ Enter the rating (0.0 to 10.0):', min_value=0.00, max_value=10.00, value=5.00, step=0.01, format="%.2f")
    num_votes_transformed = (num_votes / 10) * 1000000
    
    # Collect user input for genre
    genres = ['Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Musical', 'Romance', 'Sci-Fi']
    selected_genre = st.selectbox('🎬 Select the genre:', genres)

    st.markdown('</div>', unsafe_allow_html=True)

# Encode genre input
genre_data = {genre: 0 for genre in genres}
genre_data[selected_genre] = 1

# Prepare features for prediction
features = [1.0, budget, num_votes_transformed] + list(genre_data.values())
features = np.array(features)

# Predict button
if st.button('📊 Predict Revenue'):
    prediction = model.predict(features.reshape(1, -1))
    st.success(f'🎉 Estimated Worldwide Gross Revenue: ${prediction[0]:,.2f}')

# Additional model information 
st.markdown('## <div class="center-align">Model Information</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="center-align">
    This project is merely meant to show the capabilities of python and it's libraries. The dataset used is incomplete, and the predictions from the model do not indicate any real world events.
    </div>
    """, unsafe_allow_html=True
)
st.markdown(
    """
    <div class="center-align">
    Additionally, the prediction represents gross revenue for a films lifetime, not opening weekend. 
    </div>
    """, unsafe_allow_html=True
)

# Footer with additional branding
st.markdown("""
---
<div class="center-align">
*Developed by Mark Bozzo(https://github.com/Bozzojr)*
</div>
""", unsafe_allow_html=True)

