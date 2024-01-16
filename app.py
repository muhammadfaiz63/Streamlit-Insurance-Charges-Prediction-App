import streamlit as st
import numpy as np
import pickle
import pandas as pd  # Add this line to import pandas

# Load pre-trained model
loaded_model = pickle.load(open('model_uas.pkl', 'rb'))

# Streamlit app


def main():
    st.title("Insurance Charges Prediction App")
    st.subheader("Nama : Muhammad Faiz")
    st.subheader("NIM : 2020230065")
    st.sidebar.header("User Input")

    # Input fields for user to enter data
    age = st.sidebar.slider("Age", min_value=18, max_value=100, value=25)
    sex = st.sidebar.radio("Sex", options=['Female', 'Male'])
    bmi = st.sidebar.number_input(
        "BMI", min_value=10.0, max_value=50.0, value=25.0)
    children = st.sidebar.slider(
        "Number of Children", min_value=0, max_value=5, value=0)
    smoker = st.sidebar.radio("Smoker", options=['No', 'Yes'])

    # Convert categorical inputs to numerical values
    sex = 0 if sex == 'Female' else 1
    smoker = 1 if smoker == 'Yes' else 0

    # Display user input
    st.sidebar.subheader("User Input:")
    user_input = np.array([age, sex, bmi, children, smoker]).reshape(1, -1)
    st.sidebar.write(pd.DataFrame(user_input, columns=[
                     'Age', 'Sex', 'BMI', 'Children', 'Smoker']))

    # Make prediction using the loaded model
    charge_pred = loaded_model.predict(user_input)[0]

    # Display prediction
    st.subheader("Predicted Insurance Charges:")
    st.write(f"${charge_pred:.2f}")


if __name__ == "__main__":
    main()
