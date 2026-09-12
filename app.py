"""Streamlit front end for the Telco customer churn models.

Run it from the repository root with:  streamlit run app.py

Every algorithm trained by `train.py` is selectable in the sidebar, so the
same customer can be scored by each one for comparison.
"""

import json
from importlib.util import find_spec

import pandas as pd
import streamlit as st
from PIL import Image

from preprocessing import (
    BASE_DIR,
    INDEX_PATH,
    implausible_charges,
    load_index,
    load_model,
    prepare_features,
    unknown_values,
)

IMAGE_PATH = BASE_DIR / 'App.jpg'

# The option sets the form offers. They have to match the values the model was
# trained on, which the README lists column by column.
YES_NO = ('Yes', 'No')
YES_NO_OR_NO_INTERNET = ('Yes', 'No', 'No internet service')


@st.cache_resource
def get_index():
    return load_index()


@st.cache_resource
def get_model(slug):
    return load_model(slug)


def warn_about_unknowns(prepared, model):
    """Tell the user about category values the model was never trained on.

    The encoder ignores unrecognised categories, which silently scores the row
    as if it held the reference value. Left unreported that is a wrong answer
    with nothing to indicate it.
    """
    unknown = unknown_values(prepared, model)
    if not unknown:
        return
    lines = '\n'.join(f'- **{column}**: {", ".join(values)}'
                       for column, values in unknown.items())
    st.warning(
        'Some values in this file were never seen during training, so those '
        'rows are scored as if they held the most common value. Check the '
        'spelling against the columns listed in the README.\n\n' + lines)


def read_upload(uploaded_file):
    """Read an uploaded CSV, or explain why it could not be read.

    Returns None once a message has been shown. Without this an empty file,
    a non-CSV renamed to .csv, or a file with headers but no rows each
    reached pandas or scikit-learn raw and surfaced as a traceback.
    """
    try:
        data = pd.read_csv(uploaded_file)
    except pd.errors.EmptyDataError:
        st.error('That file is empty — it has no columns to read.')
        return None
    except (UnicodeDecodeError, pd.errors.ParserError):
        st.error('That file could not be read as CSV. Export it as plain '
                 'comma-separated text and try again.')
        return None
    if data.empty:
        st.warning('That file has column headers but no rows, so there is '
                   'nothing to predict.')
        return None
    return data


def warn_about_charges(prepared, limit=5):
    """Flag rows whose tenure and charges contradict each other."""
    problems = implausible_charges(prepared)
    if not problems:
        return
    shown = problems[:limit]
    lines = '\n'.join(f'- row {position}: {reason}' for position, reason in shown)
    if len(problems) > limit:
        lines += f'\n- ...and {len(problems) - limit} more'
    st.warning(
        'These charges do not line up with the tenure, a combination the model '
        'was never trained on, so treat the prediction with caution.\n\n' + lines)


def show_model_card(entry, is_best):
    """Sidebar summary of the selected algorithm and how it scored."""
    label = f"{entry['name']} — best" if is_best else entry['name']
    with st.sidebar.expander(f'About: {label}', expanded=False):
        st.caption(entry['family'])
        st.write(entry['blurb'])
        test = entry['test']
        left, right = st.columns(2)
        left.metric('Recall (churn)', f"{test['recall']:.3f}")
        right.metric('Precision (churn)', f"{test['precision']:.3f}")
        left.metric('F1 (churn)', f"{test['f1']:.3f}")
        right.metric('ROC AUC', f"{test['roc_auc']:.3f}")
        st.caption(
            f"Decision threshold {entry['threshold']:.3f}, tuned for F1. "
            f"Scores are on the held-out test set."
        )


def online_prediction(model, entry):
    st.info("Input data below")
    # Based on our optimal features selection
    st.subheader("Demographic data")
    seniorcitizen = st.selectbox('Senior Citizen:', YES_NO)
    dependents = st.selectbox('Dependent:', YES_NO)

    st.subheader("Payment data")
    tenure = st.slider('Number of months the customer has stayed with the company',
                       min_value=0, max_value=72, value=0)
    contract = st.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    paperlessbilling = st.selectbox('Paperless Billing', YES_NO)
    paymentmethod = st.selectbox('PaymentMethod',
                                 ('Electronic check', 'Mailed check',
                                  'Bank transfer (automatic)', 'Credit card (automatic)'))
    monthlycharges = st.number_input('The amount charged to the customer monthly',
                                     min_value=0, max_value=150, value=0)
    totalcharges = st.number_input('The total amount charged to the customer',
                                   min_value=0, max_value=10000, value=0)

    st.subheader("Services signed up for")
    multiplelines = st.selectbox("Does the customer have multiple lines",
                                 ('Yes', 'No', 'No phone service'))
    phoneservice = st.selectbox('Phone Service:', YES_NO)
    internetservice = st.selectbox("Does the customer have internet service",
                                   ('DSL', 'Fiber optic', 'No'))
    onlinesecurity = st.selectbox("Does the customer have online security",
                                  YES_NO_OR_NO_INTERNET)
    onlinebackup = st.selectbox("Does the customer have online backup",
                                YES_NO_OR_NO_INTERNET)
    techsupport = st.selectbox("Does the customer have technology support",
                               YES_NO_OR_NO_INTERNET)
    streamingtv = st.selectbox("Does the customer stream TV", YES_NO_OR_NO_INTERNET)
    streamingmovies = st.selectbox("Does the customer stream movies",
                                   YES_NO_OR_NO_INTERNET)

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
        prepared = prepare_features(features_df, 'Online')
        warn_about_charges(prepared)
        prediction = model.predict(prepared)[0]
        probability = model.predict_proba(prepared)[0][1]
        if prediction == 1:
            st.warning('Yes, the customer will terminate the service.')
        else:
            st.success('No, the customer is happy with Telco Services.')
        st.caption(
            f'Estimated probability of churn: {probability:.1%}. '
            f"{entry['name']} flags a customer above {entry['threshold']:.1%}, "
            'the threshold tuned for it during training.'
        )


def compare_all(index):
    """Score one customer with every algorithm, to see where they disagree."""
    st.subheader('Compare every algorithm')
    st.write(
        'Upload a CSV and each trained model scores the same rows, so you can see '
        'where the algorithms agree and where they part company.'
    )
    uploaded_file = st.file_uploader("Choose a file", type='csv', key='compare')
    if uploaded_file is None:
        return

    data = read_upload(uploaded_file)
    if data is None:
        return
    st.write(data.head())
    if not st.button('Compare'):
        return

    try:
        prepared = prepare_features(data, 'Batch')
    except ValueError as error:
        st.error(str(error))
        return

    warn_about_unknowns(prepared, get_model(index['best']))
    warn_about_charges(prepared)

    table = pd.DataFrame(index=range(len(prepared)))
    for entry in index['models']:
        probabilities = get_model(entry['slug']).predict_proba(prepared)[:, 1]
        table[entry['name']] = probabilities
    # How far apart the algorithms are on each customer.
    table['Spread (max - min)'] = table.max(axis=1) - table.min(axis=1)

    st.subheader('Churn probability by algorithm')
    styled = table.style.format('{:.1%}')
    # background_gradient needs matplotlib, which the app does not otherwise
    # require. It also fails lazily, at render time rather than when called,
    # so check for it upfront instead of catching the error.
    if find_spec('matplotlib') is not None:
        styled = styled.background_gradient(cmap='RdYlGn_r', axis=None)
    st.dataframe(styled)
    st.caption(
        'Each column is one algorithm. The rows with the largest spread are the '
        'customers where the choice of model actually changes the answer. Each '
        'model applies its own tuned threshold to these probabilities, so a '
        'higher number does not always mean a churn verdict.'
    )


def batch_prediction(model, entry):
    st.subheader("Dataset upload")
    uploaded_file = st.file_uploader("Choose a file", type='csv')
    if uploaded_file is None:
        return

    data = read_upload(uploaded_file)
    if data is None:
        return
    # Get overview of data
    st.write(data.head())

    if st.button('Predict'):
        try:
            prepared = prepare_features(data, 'Batch')
        except ValueError as error:
            st.error(str(error))
            return

        warn_about_unknowns(prepared, model)
        warn_about_charges(prepared)

        # Get batch prediction
        prediction_df = pd.DataFrame({
            'Predictions': model.predict(prepared),
            'Churn probability': model.predict_proba(prepared)[:, 1],
        })
        prediction_df['Predictions'] = prediction_df['Predictions'].replace({
            1: 'Yes, the customer will terminate the service.',
            0: 'No, the customer is happy with Telco Services.',
        })
        prediction_df['Churn probability'] = prediction_df['Churn probability'].map('{:.1%}'.format)

        st.subheader('Prediction')
        st.write(prediction_df)
        st.caption(
            f"{entry['name']} flags a customer above "
            f"{entry['threshold']:.1%} churn probability."
        )


def main():
    # Setting Application title
    st.title('Telco Customer Churn Prediction App')

    # Setting Application description
    st.markdown("""
     :dart:  This Streamlit app is made to predict customer churn in a fictional
    telecommunication use case. The application is functional for both online
    prediction and batch data prediction. \n
    """)

    try:
        index = get_index()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()
    except json.JSONDecodeError as error:
        st.error(f'{INDEX_PATH} is not valid JSON ({error}). '
                 'Run `python train.py` to rebuild it.')
        st.stop()

    # Setting Application sidebar default
    mode = st.sidebar.selectbox(
        "How would you like to predict?", ("Online", "Batch", "Compare all models"))

    # The models are listed best-first, as ranked by cross-validated average
    # precision during training.
    names = [entry['name'] for entry in index['models']]
    picked = st.sidebar.selectbox(
        'Algorithm', names, index=0,
        help='Every model trained by train.py, best first.',
        disabled=(mode == 'Compare all models'))
    entry = index['models'][names.index(picked)]
    show_model_card(entry, is_best=(entry['slug'] == index['best']))

    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.image(Image.open(IMAGE_PATH))

    if mode == 'Compare all models':
        compare_all(index)
        return

    try:
        model = get_model(entry['slug'])
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    if mode == "Online":
        online_prediction(model, entry)
    else:
        batch_prediction(model, entry)


if __name__ == '__main__':
    main()
