import sys 
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd 

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import FileOperationError
from src.log_config import logging

from src.utils import save_object

# Define the dataclass to hold the data transformation configuration. 
# It is a container for holding the transformed data 

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("data_repository", "preprocessor.pkl")

# Define a class for data transformation responsible for performing the actual data transformation 
class DataTransformation:
    def __init__(self):
        # Initialize the DataTransformationConfig object
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            # Separate numerical and categorical columns based on their data types 
            num_columns = [
                'Maximum Temperature ', 'Minimum Temperature', 'Dew Point ', 'Heat Index', 'Wind Chill', 'Wind Gust Kmph',
                 'Cloud Cover', 'Humidity', 'Precipitation MM', 'Pressure', 'Temperature ', 'Visibility', 'Wind Direction Degree', 'Wind speed Kmph'
                 ]
            
            cat_columns = []

            # Create a pipeline for numerical columns 
            num_pipeline = Pipeline(
                steps=[
                # Impute missing values with the median
                ("imputer",SimpleImputer(strategy="median")),
                # Scale the data using StandardScaler
                ("scaler", StandardScaler())
                ]
            ) 
            # Create a pipeline for categorical columns
            cat_pipeline = Pipeline(
                steps=[
                # Impute missing values with the most frequent value
                ("imputer", SimpleImputer(strategy = "most_frequent")),
                # One-hot encode the data
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
                # Scale the data 
                ("scaler", StandardScaler(with_mean=False))
                ]
            )
            # Create a logging message to indicate completion of above processes 
            logging.info(f"Numerical columns: {num_columns}")
            logging.info(f"Categorical columns: {cat_columns}")

            # Create a column transformer to combine the above 
            # numerical and categorical pipelines into a single pipeline
            preprocessor = ColumnTransformer(
                transformers = [
                # Apply the numerical pipeline to the numerical and categorical columns 
                ("num", num_pipeline, num_columns),
                ("cat", cat_pipeline, cat_columns)
                ]
            )

            # Create a logging message to indicate completion of above processes 
            logging.info(f"Preprocessor object: {preprocessor}")

            return preprocessor

            
        except Exception as e:
            raise FileOperationError(e, sys) 
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read the train and test data 
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Log information about reading train and test data 
            logging.info("Exited reading training and testing data")

            # Log information about obtaining the preprocessing object 
            logging.info("Obtaining preprocessing object")

            # Get the preprocessing object 
            preprocessor_obj = self.get_data_transformer_obj()

            # Define the target column name and numerical column
            # You will drop the target column 
            target_col = "Consumption"
            num_columns = [
                'Maximum Temperature ', 'Minimum Temperature', 'Dew Point ', 'Heat Index', 'Wind Chill', 'Wind Gust Kmph',
                 'Cloud Cover', 'Humidity', 'Precipitation MM', 'Pressure', 'Temperature ', 'Visibility', 'Wind Direction Degree', 'Wind speed Kmph'
                 ] 

            # Drop the target column from the input features in both training and testing sets 
            training_input_features = train_df.drop(columns = [target_col], axis = 1)
            training_target_feature = train_df[target_col]

            testing_input_features = test_df.drop(columns = [target_col], axis = 1)
            testing_target_feature = test_df[target_col]

            # Log the application of the preprocessing object on training and testing DataFrames 
            logging.info("Applying preprocessing object on training and testing dataframes")

            # Fit the preprocessing object to the training and transform the training and testing sets 
            training_input_features_array = preprocessor_obj.fit_transform(training_input_features)
            testing_input_features_array = preprocessor_obj.transform(testing_input_features)

            # Log successful transformation
            logging.info("Data transformation completed successfully.")

            # Concatenate the transformed input features with target features for both training and testing sets 
            training_arr = np.c_[training_input_features_array, np.array(training_target_feature)]
            testing_arr = np.c_[testing_input_features_array,np.array (testing_target_feature)]

            # Log the saving of the preprocessing object 
            logging.info("Saving preprocessing object")

            # Save the preprocessing object 
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            # Return the transformed train and test arrays and preprocessing object file path 
            return (
                training_arr, 
                testing_arr, 
                self.data_transformation_config.preprocessor_obj_file_path
                )
        
        
        except Exception as e:
            raise FileOperationError(e, sys)



































# import sys
# import os
# from dataclasses import dataclass

# import numpy as np
# import pandas as pd

# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder

# from src.exception import FileOperationError
# from src.log_config import logging

# from src.utils import save_object

# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path: str = os.path.join("data_repository", "preprocessor.pkl")

# class DataTransformation:
#     def __init__(self):
#         self.data_transformation_config = DataTransformationConfig()

#     def get_data_transformer_obj(self):
#         try:
#             # Get the numerical and categorical columns based on their dtypes
#             df = pd.read_csv('train.csv')  # Replace 'train.csv' with the actual training data file name
#             num_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
#             cat_columns = df.select_dtypes(include=["object"]).columns.tolist()

#             num_pipeline = Pipeline([
#                 ("imputer", SimpleImputer(strategy="median")),
#                 ("scaler", StandardScaler())
#             ])

#             cat_pipeline = Pipeline([
#                 ("imputer", SimpleImputer(strategy="most_frequent")),
#                 ("onehot", OneHotEncoder(handle_unknown="ignore")),
#                 ("scaler", StandardScaler(with_mean=False))
#             ])

#             logging.info(f"Numerical columns: {num_columns}")
#             logging.info(f"Categorical columns: {cat_columns}")

#             preprocessor = ColumnTransformer(transformers=[
#                 ("num", num_pipeline, num_columns),
#                 ("cat", cat_pipeline, cat_columns)
#             ])

#             logging.info(f"Preprocessor object: {preprocessor}")

#             return preprocessor

#         except Exception as e:
#             raise FileOperationError(e, sys)

#     def initiate_data_transformation(self, train_path, test_path):
#         try:
#             train_df = pd.read_csv(train_path)
#             test_df = pd.read_csv(test_path)

#             logging.info("Exited reading training and testing data")
#             logging.info("Obtaining preprocessing object")

#             preprocessor_obj = self.get_data_transformer_obj()

#             target_col = "Consumption"

#             training_input_features = train_df.drop(columns=[target_col], axis=1)
#             training_target_feature = train_df[target_col]

#             testing_input_features = test_df.drop(columns=[target_col], axis=1)
#             testing_target_feature = test_df[target_col]

#             logging.info("Applying preprocessing object on training and testing dataframes")

#             training_input_features_array = preprocessor_obj.fit_transform(training_input_features)
#             testing_input_features_array = preprocessor_obj.transform(testing_input_features)

#             logging.info("Data transformation completed successfully.")

#             training_arr = np.c_[training_input_features_array, np.array(training_target_feature)]
#             testing_arr = np.c_[testing_input_features_array, np.array(testing_target_feature)]

#             logging.info("Saving preprocessing object")

#             save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
#                         obj=preprocessor_obj)

#             return training_arr, testing_arr, self.data_transformation_config.preprocessor_obj_file_path

#         except Exception as e:
#             raise FileOperationError(e, sys)

# if __name__ == "__main__":
#     obj = DataTransformation()
#     train_data, test_data, preprocessor_path = obj.initiate_data_transformation("data_repository/train.csv",
#                                                                                "data_repository/test.csv")
