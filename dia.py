import os
import numpy as np
import joblib
import streamlit as st

# Get the absolute path to the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the saved model with joblib
model_path = os.path.join(BASE_DIR, 'trained_model.sav')
loaded_model = joblib.load(model_path)

st.markdown(
    """
    <style>
    .stApp {
        background: url("https://img.etimg.com/thumb/width-1200,height-900,imgsize-293924,resizemode-75,msid-73297828/magazines/panache/artificial-intellegence-for-the-win-new-ai-tool-can-predict-gestational-diabetes-risk-before-pregnancy.jpg");
        background-size: cover;
    }
    .title {
        color: lightgreen;
    }
    .stTextInput label {
        color: yellow !important;
        font-size: 1000px !important;
    }
    .result {
        color: lightgreen;
        font-size: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Creating a function for prediction
def diabetes_prediction(input_data):
    # Changing the data into a NumPy array
    input_data_as_nparray = np.asarray(input_data, dtype=float)

    # Reshaping the data since there is only one instance
    input_data_reshaped = input_data_as_nparray.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction == 0:
        return '<p class="result">Non Diabetic</p>'
    else:
        return '<p class="result">Diabetic</p>'

def main():
    # Giving a title
    st.markdown('<h1 class="title">Diabetes Prediction Web App</h1>', unsafe_allow_html=True)

    # Getting input from the user
    Pregnancies = st.text_input('No. of Pregnancies:', '0')
    Glucose = st.text_input('Glucose level:', '0')
    BloodPressure = st.text_input('Blood Pressure value:', '0')
    SkinThickness = st.text_input('Skin thickness value:', '0')
    Insulin = st.text_input('Insulin level:', '0')
    BMI = st.text_input('BMI value:', '0')
    DiabetesPedigreeFunction = st.text_input('Diabetes pedigree function value:', '0.0')
    Age = st.text_input('Age:', '0')

    # Code for prediction
    diagnosis = ''

    # Making a button for prediction
    if st.button('Predict'):
        try:
            # Convert inputs to float
            input_data = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
            diagnosis = diabetes_prediction(input_data)
        except ValueError:
            diagnosis = '<p class="result">Please enter valid numeric values for all inputs.</p>'

    st.markdown(diagnosis, unsafe_allow_html=True)

if __name__ == '__main__':
    main()