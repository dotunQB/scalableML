# Script to train machine learning model.
import pandas as pd
import pickle
import csv
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
from joblib import load

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


def slice_data_run(model_input, test_df, label, encoder, lb):
    """ Function for calculating ml run on slices of the census dataset for salary.
    model_input: Any
            Picked model to be used for inference
    test_df: pd.DataFrame
            input dataframe to be utilized for run
    label:
        label class for data to be sliced
    encoder:
        encoder generated during training
    lb:
        binarizer generated during training
    """
    csvFile = open("slice_output.txt", "w", newline="", encoding='utf-8')
    # csvFile.truncate()
    csvWriter = csv.writer(csvFile)
    # label = 'race'
    for cls in test_df[label].unique():
        test_slice = test_df[test_df[label] == cls]
        X_test, y_test, encoder, lb = process_data(
            test_slice, categorical_features=cat_features, label="salary", training=False,
            encoder=encoder, lb=lb
        )
        preds = inference(model_input, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, preds)

        # Print out metrics
        # print(f"Here is the model output for Race: {race}")
        csvWriter.writerow([f"Here is the model output for {label}: {cls}"])
        csvWriter.writerow(["Prediction for current run is: "])
        csvWriter.writerow(preds)
        csvWriter.writerow([f"Recall for current run is: "])
        csvWriter.writerow([recall])
        csvWriter.writerow([f"Fbeta for current run is: "])
        csvWriter.writerow([fbeta])
    csvFile.close()


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
pickle.dump(model, open("model_out.pkl", "wb"))
pickle.dump(encoder, open("encoder_out.pkl", "wb"))
pickle.dump(lb, open("lb_out.pkl", "wb"))
# Print out metrics
print("prediction for input is: ", preds)
print("precision is: ", precision)
print("recall is: ", recall)
print("fbeta is: ", fbeta)

# return encoder, lb
model_input = load(open('./model_out.pkl', 'rb'))
# run on slice of data
slice_data_run(model_input=model_input, test_df=test, label='education', encoder=encoder, lb=lb)
