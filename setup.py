# Import libraries 
from setuptools import find_packages, setup
from typing import List

# Define a constant for the hyphen-e-dot string 
HYPHEN_E_DOT_REQUIREMENT = '-e .'

# Create a function that takes a file path as input and returns a list of requirements from the requirements.txt file 
def get_requirements(file_path:str) -> List[str]:
    # Create an empty list that will store the requirements 
    requirements = []

    # Open the requirements.txt file 
    with open(file_path) as file_obj:
        # Iterate over each line in the file
        for line in file_obj:
            # Remove the newline character at the end of each line 
            requirements.append(line.replace("\n", ""))

    # Remove the '-e .' if present in the list 
    if HYPHEN_E_DOT_REQUIREMENT in requirements:
        requirements.remove(HYPHEN_E_DOT_REQUIREMENT)

    # Return the requirements 
    return requirements
    

setup(
    name="Home Electricity Monitoring System",
    version="0.1",
    packages=find_packages(),
    author = "Western",
    author_email="minichworks@gmail.com",
    description="A Machine Learning Project for Predicting Electricity Consumption",
    install_requires= get_requirements('requirements.txt')      
)