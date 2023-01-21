# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- This model was developed by Dotun Opasina on Jan 20 2023
- The model is a simple logistic regression on census data to predict salary
- The dataset used is US population census data
## Intended Use
- This model intended purpose is solely for experiment use
- As the dataset was collected in 1994, this data should not be used in today's World.
## Training Data
- The training data is based on 80% of the dataset
## Evaluation Data
- The model is evaluated on 20% of the dataset
## Metrics
_Please include the metrics used and your model's performance on those metrics._
On the entire data, without slicing out any specific columns, the performance is as follows:
- Precision is: 73.31% 
- Recall is:  26.90%
- Fbeta is: 39.36%
## Ethical Considerations
- As mentioned earlier, this model operates on outdated data so it output is for showcasing purposes
- The outdated data has about 48k observations, this might not be a representative size of the groups present
## Caveats and Recommendations
- Additional bias and fairness evaluation need to be tested on the data source
- Performance on appropriate validation of the model such as k-fold cross validation could help increase the robustness of the model
