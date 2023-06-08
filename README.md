# Project Description:
Corona Virus 19 is an airborne-virus that affected the world during 2020-2021, especially countries with a vas low income population. Covid-19 virus mainly affected elder and people with chronic illneses such as diabetes, respiratory diseases, and cardiovascular issues. 
We would be looking at pre-conditions and how it affectes patients that were tested for covid-19.
This data set was provided from the government of Mexico, contains anonymous information of patients who were tested of covid during 2020 and 2021
# Project Goals
The main goal of this project is to build a machine learning model that, given a Covid-19 patient's current symptom, status, and medical history, will predict whether the patient is in high-risk, medium-risk or low-risk.

# Initial Hypotheses and Questions about the Data:
1. Does being a smoker have a correlation with covid positive tests?
2. Patient who died have a direct correlation with covid positive tests?
3. Patients with copd have a correlation with covid positive tests?
4. Patient with heart problems have a correlation with covid positive tests?
5. Does hospitalizations have a correlation with covid positive tests?
6. Does ICU commitment have a correlation with covid positive tests?
7. Does hypertension have a correlation with covid risk?

# Data dictionary:
![Alt text](https://github.com/Chellyann-moreno/covid19-project/blob/main/Data%20Dictionary.png)


For more information please visit: https://www.kaggle.com/datasets/meirnizri/covid19-dataset

# Project Planning, Layout of Science Pipeline:
1. Project Planning- During this process we asked ourselves important questions about the project. Research and choose the data set for this project.
Data planning will be shown in this readme.
2. Data acquisition- We would be acquiring the data from [Kaggle.com](https://www.kaggle.com/datasets/meirnizri/covid19-dataset). Raw data would be downloaded and "covid_data.csv" has been created which would be use to pull the data during this project.
3. Data preparation- The covid data would be clean and prepared for exploration. Columns were listed along with data types. columns with incorrect data types were transformed, columns were created to encode a int/float instead of a string, and columns that are not to be used were dropped. Columns were renamed and nulls were handled accordingly.
4. Data exploration- During the data exploration we would visualize and answer our questions and hypotheses. We would be using statistical tests and plots, to help proving our hypotheses and understand the various risk level of patients who were tested for covid.
5. Data modeling- We would be using the most important features(15-20) to check the accuracy and recall scores. By looking on how to predict patients risk level before being tested for covid, and be able to provide the care and next steps. 
6. Data Delivery- We would be using a jupyter notebook, where we would be showing our visualizations, questions and findings.
# Instructions on How to Reproduce this Project:
For an user to succesfully reproduce this project, they must succesfully download the dataset from [Kaggle.com](https://www.kaggle.com/datasets/meirnizri/covid19-dataset). User must have proper wrangle.py explore.py, model.py, and final notebook. All documents must be downloaded in the same repository/folder to be able to run it successfully. 
Once all files are download, user may run the final_report notebook.

# Executive Summary:
