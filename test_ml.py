import pytest
# TODO: add necessary import
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from ml.model import (
    inference,
    train_model,
)

def check_columns():
    #Check if number of Columns are the same number of columns coming from the Source File
    path = './data/census.csv'
    data = pd.read_csv(path)
    columns = 15
    columns_read = data.shape[1]
    assert columns == columns_read


def model_inference_test():    
    #Testing the inference of the current model    
    X = np.random.rand(20, 5)
    y = np.random.randint(2, size=30)
    model_test = train_model(X, y)
    y_predictions = inference(model_test, X)
    assert y.shape == y_predictions.shape



def algorithm_test():
    #Tests for using the RandomForestClassifier algorithm
    Xtrain = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    ytrain = np.array([0, 1, 0, 1])  
    model_algorithm = train_model(Xtrain, ytrain)
    assert isinstance(model_algorithm, RandomForestClassifier), "RandomForestClassifier is not used for the model"
