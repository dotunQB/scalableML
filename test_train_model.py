import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, inference

@pytest.fixture(scope="session")
def cat_feat():
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    return cat_features

@pytest.fixture(scope="session")
def data():
    local_path = "./data/census.csv"
    df = pd.read_csv(local_path, low_memory=False)
    return df

def test_train_test_split(data):
    train, test = train_test_split(data, test_size=0.20)
    assert len(train) < len(data)
    assert len(test) < len(data)


def test_process_data(data, cat_feat):
    train, test = train_test_split(data, test_size=0.20)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_feat, label="salary", training=True
    )
    assert len(X_train) == len(train)
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_feat, label="salary", training=False,
        encoder=encoder, lb=lb
    )
    assert len(X_test) == len(test)

def test_train_model(data, cat_feat):
    train, test = train_test_split(data, test_size=0.20)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_feat, label="salary", training=True
    )
    # Process the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_feat, label="salary", training=False,
        encoder=encoder, lb=lb
    )
    # Train and save a model.
    model_out = train_model(X_train, y_train)
    preds = inference(model_out, X_test)
    assert len(preds) == len(X_test)
