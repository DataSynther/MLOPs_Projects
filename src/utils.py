## Usually holds utility functions that are used in multiple places in the codebase
#Import necessary libraries

import sys
import os

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
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