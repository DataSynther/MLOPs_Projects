## Contains all codes related to data transformation

# Importimng necessary libraries
import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging


# Defining a class for data transformation
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transfromation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):

        '''
        This function is responsible for data transformation for differnet types of data 

        '''

        try:
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy= "median")),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")), #Replpace missing values with mode 
                    ("One_hot_encoder", OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info (f"Numerical columns: {numerical_columns}")
            logging.info (f"Categorical columns: {categorical_columns}")

            preprocessor =ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            logging.info("Preprocessor object created successfully")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self, train_path, test_path):

        try:
            # Read the data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and Test data reading is successful!!") 

           #Open the preprocessor object file
            logging.info("Opening the preprocessor object file")
            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = 'math_score'
            numerical_columns = ['reading_score', 'writing_score']

            input_feature_train_df = train_df.drop(target_column_name, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(target_column_name, axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessor object on train data and test data")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]
            
            logging.info(f"Saved preprocessing object")
            
            #usded to save the preprocessing object as a pkl file
            save_object(
                file_path = self.data_transfromation_config.preprocessor_obj_file,
                obj = preprocessing_obj
            )
            
            return(
                train_arr,
                test_arr, 
                self.data_transfromation_config.preprocessor_obj_file
            )



        except Exception as e:
            raise CustomException(e,sys)