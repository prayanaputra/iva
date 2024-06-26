import pickle
import streamlit as st
import numpy as np

# Load the model
model = pickle.load(open('diabetes_model.sav', 'rb'))  # Ensure to replace 'diabetes_model.sav' with your actual model file

# Set page config
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Define the front end interface
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    h1 {
        color: #ff4b4b;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Diabetes Prediction App')

st.markdown(
    """
    ### Please enter the following details:
    """
)

# Layout for input fields
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
    glucose = st.number_input('Glucose', min_value=0, max_value=200, value=0)
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=150, value=0)
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=0)

with col2:
    insulin = st.number_input('Insulin', min_value=0, max_value=900, value=0)
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=0.0, format="%.1f")
    diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.0, format="%.3f")
    age = st.number_input('Age', min_value=0, max_value=120, value=0)

# Make predictions
if st.button('Predict'):
    input_data = (
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, diabetes_pedigree_function, age
    )
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)

    st.markdown(
        """
        ### Prediction Result:
        """
    )

    if prediction[0] == 0:
        st.success('The person is not diabetic.')
    else:
        st.error('The person is diabetic.')

# Footer
st.markdown(
    """
    <hr>
    <p style="text-align:center;">with Kelompok 1 - 21S1SI-Machine2(SI163) </p>
    """,
    unsafe_allow_html=True
)
