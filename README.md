# INSE6220 - FinalProject - Fall 2023
## Body Mass Index Classification using Principal Component Analysis and Machine Learning 
## Alexander Ireland – 40292168
_Special Note: Python code, created in collaboration with ChatGTP_


## About
The project contains 2 python files. Run them in the order shown below.
### dataProcessingAndPCA.py
_**Input:** bodyfat.csv_  
_**Output:** PCA_output.csv_  
The code takes as input the raw dataset file bodyfat.csv and performs and data analysis and then generates the plots related to the analysis and PCA. The file also outputs a datafile PCA_output.csv that contains the first two Principal Components and the class feature as well. That csv is used as the input to machineLearning.py
### machineLearning.py
_**Input:** PCA_output.csv_  
The code takes as input PCA_output.csv(that was generated by dataProcessingAndPCA.py) and performs and visualizes the various machine learning tasks used in the report
