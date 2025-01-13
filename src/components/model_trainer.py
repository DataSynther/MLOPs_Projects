## This script will contain all the functions related to training the model
# Importing necessary libraries
#Basic Imports
import os
import sys
# import dill
from dataclasses import dataclass

#Modelling Imports
from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

#Custom Imports 
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evalute_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
       
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            #Model dictionary
          
            # Define the models and parameters
            models = {
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=0),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "XGBRegressor": {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [100, 200, 300],
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'iterations': [100, 200, 300],
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                },
            }

       
            #Model training and evaluation
            model_report = evalute_model(X_train=X_train, y_train=y_train, X_test = X_test, y_test= y_test, models=models,param =params)
            logging.info(f"Model training and evaluation is completed")

            #To get best model score and model name sorted by score
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ] 
            best_model = models[best_model_name]

            #As a basic criteria of model fitting we will use the prediction has to be atleast 60% accurate
            if best_model_score<0.6:
                raise CustomException("No Best model found",sys)
            else:     
                logging.info(f"Best model is {best_model_name} with score {best_model_score}")

                #Training the best model
                # preprocessing_obj = dill.load(open(preprocessor_path, "rb"))
                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )

                predicted = best_model.predict(X_test)

                #Model evaluation
                r2 = r2_score(y_test,predicted)
                return r2

        except Exception as e:
            raise CustomException(e,sys)