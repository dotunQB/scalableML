# Scalable ML Live Deployment Udacity Submission
# Environment Set up
* Download and install conda if you don’t have it already.
    * Use the supplied requirements file to create a new environment, or
    * conda create -n [envname] "python=3.8" scikit-learn pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge
    * Install git either through conda (“conda install git”) or through your CLI, e.g. sudo apt-get git.
    * Install the needed files requirements text using pip install -r requirements

## Repositories
* Clone output from: https://github.com/dotunQB/scalableML
# Data
* Get data from the data folder present in the repo
* Data is called census.csv

# Model
* Model written using logistic regression to predict the salary based on census data

# API Creation
*  A rest end point has been created, to run locally try the following: 
  * uvicorn main:app --reload 

# API Deployment
* This model has been deployed live on render can be found in the following link: https://render-scalableml.onrender.com