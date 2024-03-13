import os 
import sys
import pickle 
import dill

from src.exception import FileOperationError


# Define a function to save an object in a file 
def save_object(file_path, obj):
    try:
        # Extract the directory path from the given file path 
        # Get the directory path of the file 
        dir_path = os.path.dirname(file_path)
        # If the directory path does not exist, create it 
        os.makedirs(dir_path, exist_ok = True)
        # Save the object in the file 
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            #dill.dump(obj, file_obj)
    # Handle exception that may occur
    except Exception as e:
        raise FileOperationError(e, sys)