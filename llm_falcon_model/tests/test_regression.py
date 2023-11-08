import joblib


def test_regression():
    reference = joblib.load('ref-falcon-7b.pkl')
    print(len(reference))