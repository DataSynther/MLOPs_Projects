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
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Garadient Boosting": GradientBoostingRegressor(),
                "Liner Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            #Model training and evaluation
            model_report:dict = evalute_model(X_train=X_train, y_train=y_train, X_test = X_test, y_test= y_test, models=models)
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