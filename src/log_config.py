import logging 
import os 
from datetime import datetime 

# Define the log file name using the current date and time 
log_file_name = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create the path to the log file inside the logs directory in the current working directory
log_file_path = os.path.join (os.getcwd(), "logs", log_file_name)

# Create the logs directory if it does not exist 
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Configure the logging module providing basic settings 
logging.basicConfig(
    filename=log_file_path,   # Specify the log file path 
    level=logging.INFO,       # Set the logging level to INFO meaning only INFO level messages and above will be raised 
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Define the log message format
)

# # Testing the logger 
# if __name__ == "__main__":
#     # Log information message
#     logging.info("Logging into the System")