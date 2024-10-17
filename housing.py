# Import necessary libraries
import streamlit as st
import numpy as np
import pickle

# Load the pre-trained model
with open('model_lin.pkl', 'rb') as file:  
    model = pickle.load(file)

# Streamlit app interface
st.title(' 🏡 House Price Prediction App')

st.write("""
### Enter the details below to predict the house price:
""")

# Get user input for each feature
income = st.number_input('Average Area Income', min_value=0.0, value=50000.0)
house_age = st.number_input('Average Area House Age', min_value=0.0, value=5.0)
num_rooms = st.number_input('Average Area Number of Rooms', min_value=1.0, value=6.0)
num_bedrooms = st.number_input('Average Area Number of Bedrooms', min_value=1.0, value=3.0)
population = st.number_input('Area Population', min_value=0.0, value=20000.0)

# Button to predict
if st.button('Predict House Price'):
    # Prepare input data for prediction
    input_data = np.array([[income, house_age, num_rooms, num_bedrooms, population]])

    # Predict the price using the loaded model
    predicted_price = model.predict(input_data)

    # Display the prediction
    st.success(f'The predicted house price is: ${predicted_price[0]:,.2f}')
