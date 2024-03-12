import os 
import sys
from dataclasses import dataclass

import pandas as pd 
from sklearn.model_selection import train_test_split 

from src.exception import FileOperationError
from src.log_config import logging



# Define a dataclass that will hold data ingestion configuration
# This is a container for holding configuration data such as file path for training, testing, and raw data

@dataclass
class DataIngestionConfig:
    """
    DataConfig is a container for holding configuration data such as file path for training, testing, and raw data
    """
    # Define the default file path for train, test and raw data 
    # We indicate that training data will be stored in a folder named 'data_repository' and file name will be train.csv
    train_data_path:str = os.path.join ("data_repository", "train.csv")
    # The path of testing data
    test_data_path:str = os.path.join ("data_repository", "test.csv")
    # The path of raw data
    raw_data_path:str = os.path.join ("data_repository", "raw.csv")


# Define a dataclass for data input. This will perform the actual data importation
class DataImportation:
    # Create a constructor to initialize the data importation object 
    def __init__(self, config: DataIngestionConfig):
        # Create an instance of the DataImportationConfig to store configuration 
        # The import_config attribute of the data importation class is used to store the configuration for data dat input process 
        # This configuration will include the paths of the train, test and raw data 
        self.config = config 

    # Method to initiate data importation process 
    def initiate_data_importation(self):
        # Log message to indicate start of data importation process 
        logging.info("Entered the data importation")
        try:
            # Read the dataset as a DataFrame
            df = pd.read_csv(r'notebook\data\Home Electricity Consumption.csv')
            # Log message to indicate successful reading of the dataframe 
            logging.info("Successfully read the dataframe")

            # Create a directory for train data if it does not exist
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True) 

            # Save the dataframe to to the raw data path 
            df.to_csv(self.config.raw_data_path, index=False, header=True)
            # Log message to indicate end of data importation process and saving raw data 
            logging.info("Exited the data importation")

            # Log message to commence splitting to training and testing set 
            logging.info("Train test splitting has started ")
            # Split dataset into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.25, random_state=42)

            # Save the train set to the train data path 
            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            # Save the test set to the test data path 
            test_set.to_csv(self.config.test_data_path, index=False, header=True)
            # Log message to indicate end of data importation process 
            logging.info("Exited the data splitting process.")

            # Return the paths to the train and test data 
            return (
                self.config.train_data_path,
                self.config.test_data_path
            )

        except Exception as e:
            # Log the exception
            #logging.error(e)
            # Raise the exception
            raise FileOperationError(e, sys)
        
if __name__ == "__main__":
    config = DataIngestionConfig()
    obj=DataImportation(config)
    obj.initiate_data_importation()




    

