"""Streamlit front end for the Telco customer churn model.

Run it from the repository root with:  streamlit run app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from PIL import Image

from preprocessing import preprocess

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / 'notebook' / 'model.sav'
IMAGE_PATH = BASE_DIR / 'App.jpg'


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def online_prediction(model):
    st.info("Input data below")
    # Based on our optimal features selection
    st.subheader("Demographic data")
    seniorcitizen = st.selectbox('Senior Citizen:', ('Yes', 'No'))
    dependents = st.selectbox('Dependent:', ('Yes', 'No'))

    st.subheader("Payment data")
    tenure = st.slider('Number of months the customer has stayed with the company', min_value=0, max_value=72, value=0)
    contract = st.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    paperlessbilling = st.selectbox('Paperless Billing', ('Yes', 'No'))
    paymentmethod = st.selectbox('PaymentMethod', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    monthlycharges = st.number_input('The amount charged to the customer monthly', min_value=0, max_value=150, value=0)
    totalcharges = st.number_input('The total amount charged to the customer', min_value=0, max_value=10000, value=0)

    st.subheader("Services signed up for")
    multiplelines = st.selectbox("Does the customer have multiple lines", ('Yes', 'No', 'No phone service'))
    phoneservice = st.selectbox('Phone Service:', ('Yes', 'No'))
    internetservice = st.selectbox("Does the customer have internet service", ('DSL', 'Fiber optic', 'No'))
    onlinesecurity = st.selectbox("Does the customer have online security", ('Yes', 'No', 'No internet service'))
    onlinebackup = st.selectbox("Does the customer have online backup", ('Yes', 'No', 'No internet service'))
    techsupport = st.selectbox("Does the customer have technology support", ('Yes', 'No', 'No internet service'))
    streamingtv = st.selectbox("Does the customer stream TV", ('Yes', 'No', 'No internet service'))
    streamingmovies = st.selectbox("Does the customer stream movies", ('Yes', 'No', 'No internet service'))

    data = {
        'SeniorCitizen': seniorcitizen,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phoneservice,
        'MultipleLines': multiplelines,
        'InternetService': internetservice,
        'OnlineSecurity': onlinesecurity,
        'OnlineBackup': onlinebackup,
        'TechSupport': techsupport,
        'StreamingTV': streamingtv,
        'StreamingMovies': streamingmovies,
        'Contract': contract,
        'PaperlessBilling': paperlessbilling,
        'PaymentMethod': paymentmethod,
        'MonthlyCharges': monthlycharges,
        'TotalCharges': totalcharges,
    }
    features_df = pd.DataFrame.from_dict([data])

    st.write('Overview of input is shown below')
    st.dataframe(features_df)

    if st.button('Predict'):
        prediction = model.predict(preprocess(features_df, 'Online'))
        if prediction[0] == 1:
            st.warning('Yes, the customer will terminate the service.')
        else:
            st.success('No, the customer is happy with Telco Services.')


def batch_prediction(model):
    st.subheader("Dataset upload")
    uploaded_file = st.file_uploader("Choose a file", type='csv')
    if uploaded_file is None:
        return

    data = pd.read_csv(uploaded_file)
    # Get overview of data
    st.write(data.head())

    if st.button('Predict'):
        try:
            preprocess_df = preprocess(data, 'Batch')
        except ValueError as error:
            st.error(str(error))
            return

        # Get batch prediction
        prediction = model.predict(preprocess_df)
        prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
        prediction_df = prediction_df.replace({
            1: 'Yes, the customer will terminate the service.',
            0: 'No, the customer is happy with Telco Services.',
        })

        st.subheader('Prediction')
        st.write(prediction_df)


def main():
    # Setting Application title
    st.title('Telco Customer Churn Prediction App')

    # Setting Application description
    st.markdown("""
     :dart:  This Streamlit app is made to predict customer churn in a fictional telecommunication use case.
    The application is functional for both online prediction and batch data prediction. \n
    """)

    # Setting Application sidebar default
    add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.image(Image.open(IMAGE_PATH))

    model = load_model()
    if add_selectbox == "Online":
        online_prediction(model)
    else:
        batch_prediction(model)


if __name__ == '__main__':
    main()
