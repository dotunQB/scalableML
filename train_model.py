# Script to train machine learning model.
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# load data from census csv
data = pd.read_csv("./data/census.csv")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
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

def general_pipeline(data, cat_features, path_name):
    """
    Function to run the general modelling pipeline to avoid repetition
    data : pd.DataFrame
        Input Dataframe
    cat_features: array
        Categorical columns label for preprocessing
    path_name: str
        Output of where to save model output
    """
    train, test = train_test_split(data, test_size=0.20)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    # Process the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lb
    )
    # Train model and run evaluation on inference
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print("saving output")
    # Save model from path
    pickle.dump(model, open(path_name, "wb"))
    # Print out metrics
    print("precision is: ", precision)
    print("recall is: ", recall)
    print("fbeta is: ", fbeta)

    return encoder, lb


# run on general data without slices
general_pipeline(data, cat_features, "model_out.pkl")

### Run pipeline on slices of data, using categorical features
slice_col=cat_features + ["salary"]
sliced_data = data[slice_col]
encoder, lb = general_pipeline(sliced_data, cat_features, "model_slice.pkl")

pickle.dump(encoder, open("encoder_out.pkl", "wb"))
pickle.dump(lb, open("lb_out.pkl", "wb"))