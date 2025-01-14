## Usually holds utility functions that are used in multiple places in the codebase
#Import necessary libraries

import sys
import os

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from src.logger import logging

def save_object(file_path,obj):
    '''
    This function is responsible for saving the object to the specified file path
    
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open (file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
def evalute_model(X_train, X_test, y_train, y_test, models,param):
    
    '''
    This function is responsible for training the model and evaluating the model performance
    
    '''
    try:
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]

            para = param[model_name]
            logging.info( f"Printing model parameter value as {param}")
            
            #Train the model
            gs = GridSearchCV(estimator=model, param_grid=para, cv=5)
            logging.info( f"Printing model Grid Seach as {gs}")
            gs.fit(X_train, y_train)
            
            # Set the best parameters to the model
            best_params = gs.best_params_
            model.set_params(**best_params)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            #Evaluate the model
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            #report creation
            report[model_name] = test_model_score

        return report
        
    except Exception as e:
        raise CustomException(e,sys)


def load_object(file_path):
    '''
    This function is responsible for loading the object from the specified file path
    
    '''
    try:
        with open(file_path,"rb") as file_obj:
            obj = dill.load(file_obj)
            return obj
    except Exception as e:
        raise CustomException(e,sys)