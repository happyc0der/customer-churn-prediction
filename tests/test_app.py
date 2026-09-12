"""Tests for the Streamlit app.

`read_upload` is exercised directly; the rest goes through Streamlit's own
AppTest harness, which runs the script without a browser.
"""

import io

import pytest
from streamlit.testing.v1 import AppTest

import app
from preprocessing import known_categories


def upload(text, encoding='utf-8'):
    return io.BytesIO(text.encode(encoding) if isinstance(text, str) else text)


HEADER = ','.join([
    'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'])


# Each of these used to surface as a traceback in the browser.
def test_empty_file_is_refused():
    assert app.read_upload(upload('')) is None


def test_headers_without_rows_are_refused():
    assert app.read_upload(upload(HEADER + '\n')) is None


def test_unreadable_bytes_are_refused():
    assert app.read_upload(io.BytesIO(b'\xe8\x00\xff\xfe rubbish')) is None


def test_a_real_file_is_read(root):
    with open(root / 'data' / 'batch_churn.csv', 'rb') as handle:
        data = app.read_upload(handle)
    assert data is not None and len(data) == 5


@pytest.fixture
def running_app(root):
    at = AppTest.from_file(str(root / 'app.py'), default_timeout=60)
    at.run()
    return at


def test_app_starts_without_exception(running_app):
    assert not running_app.exception


def test_sidebar_lists_every_model(running_app, index):
    algorithms = running_app.sidebar.selectbox[1]
    assert algorithms.label == 'Algorithm'
    assert algorithms.options == [entry['name'] for entry in index['models']]
    assert algorithms.value == index['models'][0]['name']


def test_prediction_modes_are_offered(running_app):
    assert running_app.sidebar.selectbox[0].options == [
        'Online', 'Batch', 'Compare all models']


def test_online_prediction_reports_a_verdict_and_probability(running_app):
    running_app.button[0].click().run()
    assert not running_app.exception
    said = ' '.join(w.value for w in running_app.warning) + \
           ' '.join(s.value for s in running_app.success)
    assert 'terminate the service' in said or 'happy with Telco' in said
    assert any('probability of churn' in c.value for c in running_app.caption)


def test_switching_algorithm_switches_the_model_card(running_app, index):
    last = index['models'][-1]['name']
    running_app.sidebar.selectbox[1].set_value(last).run()
    assert not running_app.exception
    assert running_app.sidebar.selectbox[1].value == last


def in_batch_mode(app_test):
    app_test.sidebar.selectbox[0].set_value('Batch').run()
    return app_test


def messages(app_test):
    return ' '.join([w.value for w in app_test.warning]
                    + [e.value for e in app_test.error])


# These drive the real upload path. Testing read_upload on its own is not
# enough: it catches nothing if the upload path stops calling it.
def test_app_refuses_a_file_with_no_rows(running_app):
    at = in_batch_mode(running_app)
    at.file_uploader[0].upload('empty.csv', (HEADER + '\n').encode())
    at.run()
    assert not at.exception
    assert 'no rows' in messages(at)


def test_app_refuses_an_unreadable_file(running_app):
    at = in_batch_mode(running_app)
    at.file_uploader[0].upload('rubbish.csv', b'\xe8\x00\xff\xfe not text')
    at.run()
    assert not at.exception
    assert 'could not be read as CSV' in messages(at)


def test_app_refuses_a_file_with_the_wrong_columns(running_app):
    at = in_batch_mode(running_app)
    at.file_uploader[0].upload('wrong.csv', b'a,b,c\n1,2,3\n')
    at.run()
    at.button[0].click().run()
    assert not at.exception
    assert 'missing required columns' in messages(at)


def test_app_predicts_from_a_good_file(running_app, root):
    at = in_batch_mode(running_app)
    at.file_uploader[0].upload(
        'batch.csv', (root / 'data' / 'batch_churn.csv').read_bytes())
    at.run()
    at.button[0].click().run()
    assert not at.exception
    assert messages(at) == '', f'a clean file should not warn: {messages(at)}'
    assert any('churn probability' in c.value.lower() for c in at.caption)


def test_app_warns_about_an_unrecognised_category(running_app, root):
    text = (root / 'data' / 'batch_churn.csv').read_text()
    at = in_batch_mode(running_app)
    at.file_uploader[0].upload(
        'typo.csv', text.replace('Month-to-month', 'Month to month', 1).encode())
    at.run()
    at.button[0].click().run()
    assert not at.exception
    assert 'never seen during training' in messages(at)
    assert 'Month to month' in messages(at)


def in_compare_mode(app_test):
    app_test.sidebar.selectbox[0].set_value('Compare all models').run()
    return app_test


def test_compare_mode_refuses_a_file_with_no_rows(running_app):
    at = in_compare_mode(running_app)
    at.file_uploader[0].upload('empty.csv', (HEADER + '\n').encode())
    at.run()
    assert not at.exception
    assert 'no rows' in messages(at)


def test_compare_mode_refuses_an_unreadable_file(running_app):
    at = in_compare_mode(running_app)
    at.file_uploader[0].upload('rubbish.csv', b'\xe8\x00\xff\xfe not text')
    at.run()
    assert not at.exception
    assert 'could not be read as CSV' in messages(at)


def test_compare_mode_scores_every_algorithm(running_app, root, index):
    at = in_compare_mode(running_app)
    at.file_uploader[0].upload(
        'batch.csv', (root / 'data' / 'batch_churn.csv').read_bytes())
    at.run()
    at.button[0].click().run()
    assert not at.exception

    table = at.dataframe[-1].value
    for entry in index['models']:
        assert entry['name'] in table.columns
    assert 'Spread (max - min)' in table.columns


# The form's options and the encoder's vocabulary have to stay in step. If a
# widget offers a spelling the model never saw, the app silently scores that
# choice as the reference category -- the same silent wrongness the unknown
# value warning exists to catch, except here it comes from our own UI.
FORM_LABEL_TO_COLUMN = {
    'Senior Citizen:': 'SeniorCitizen',
    'Dependent:': 'Dependents',
    'Contract': 'Contract',
    'Paperless Billing': 'PaperlessBilling',
    'PaymentMethod': 'PaymentMethod',
    'Does the customer have multiple lines': 'MultipleLines',
    'Phone Service:': 'PhoneService',
    'Does the customer have internet service': 'InternetService',
    'Does the customer have online security': 'OnlineSecurity',
    'Does the customer have online backup': 'OnlineBackup',
    'Does the customer have technology support': 'TechSupport',
    'Does the customer stream TV': 'StreamingTV',
    'Does the customer stream movies': 'StreamingMovies',
}


def test_form_offers_exactly_what_the_model_was_trained_on(running_app, best_model):
    vocabulary = known_categories(best_model)

    checked = 0
    for box in running_app.selectbox:
        column = FORM_LABEL_TO_COLUMN.get(box.label)
        if column is None:
            continue
        assert set(box.options) == set(vocabulary[column]), \
            f'{column}: form offers {sorted(box.options)}, model knows {sorted(vocabulary[column])}'
        checked += 1

    assert checked == len(FORM_LABEL_TO_COLUMN), 'a form widget went missing'
