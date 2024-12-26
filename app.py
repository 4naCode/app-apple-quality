import streamlit as st
import numpy as np
import joblib

# Load the pre-trained model
model = joblib.load('rf_model.pkl')  # Replace with your actual model path

# Set the title of the app
# Set the title of the app and center it
st.markdown("<h1 style='text-align: center;'>Apple Quality Prediction</h1>", unsafe_allow_html=True)


# Create sliders and inputs for the features
size = st.slider('Size', min_value=-10.0, max_value=10.0, value=0.0, step=0.01)
weight = st.slider('Weight', min_value=-10.0, max_value=10.0, value=0.0, step=0.01)
sweetness = st.slider('Sweetness', min_value=-10.0, max_value=10.0, value=0.0, step=0.01)
crunchiness = st.slider('Crunchiness', min_value=-10.0, max_value=10.0, value=0.0, step=0.01)
juiciness = st.slider('Juiciness', min_value=-10.0, max_value=10.0, value=0.0, step=0.01)
ripeness = st.slider('Ripeness', min_value=-10.0, max_value=10.0, value=0.0, step=0.01)
acidity = st.slider('Acidity', min_value=-10.0, max_value=10.0, value=0.0, step=0.01)


# When the user clicks the Predict button
if st.button('Predict'):
    # Create an array of the input features
    input_features = np.array([[size, weight, sweetness, crunchiness, juiciness, ripeness, acidity]])

    # Make a prediction using the loaded model
    prediction = model.predict(input_features)


     # Convert prediction result to "Good" or "Bad"
    result = 'Good' if prediction[0] == 1 else 'Bad'

    # Display the prediction
    st.write(f"Predicted Outcome: ")
     # Display the prediction in the center with larger font
    st.markdown(f"<h3 style='text-align: center; font-size: 50px;'>{result}</h3>", unsafe_allow_html=True)
