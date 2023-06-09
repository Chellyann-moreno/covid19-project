# Project Description:
The Corona Virus 19, also known as COVID-19, is an airborne virus that significantly impacted the global population between 2020 and 2021. It particularly affected countries with a large low-income population. Elderly individuals and those with pre-existing chronic illnesses such as diabetes, respiratory diseases, and cardiovascular issues were especially susceptible to the virus.

In this project, we aim to analyze the impact of pre-existing conditions on patients who tested positive for COVID-19. The dataset used for this analysis was obtained from the government of Mexico and contains anonymized information of individuals who underwent COVID-19 testing during the period spanning 2020 to 2021.

# Project Goals
The primary objective of this project is to develop a robust machine learning model capable of accurately classifying the risk level (high, medium, or low) for Covid-19 patients based on their current symptoms, medical status, and comprehensive medical history. By leveraging advanced predictive modeling techniques, we aim to enhance risk assessment capabilities, aiding healthcare professionals in making informed decisions and providing personalized care for patients affected by Covid-19.

The project goal is to develop a model that accurately predicts the risk level of COVID-19 patients, enabling hospitals and clinics to enhance their preparedness and assessment. This will ensure the availability of essential resources, including tools, equipment, and staff, to effectively manage future patients, irrespective of their test results or patient volume.

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
Instructions: 
1. Access the COVID-19 Project repository on GitHub.
2. Click on the "Code" button and select "Download ZIP" to download the entire repository to your computer.Extract the downloaded    ZIP file to a directory of your choice. Or you may copy the SSH code onto your terminal. 
3. Go to [Kaggle.com](https://www.kaggle.com/datasets/meirnizri/covid19-dataset) and download the COVID-19 dataset in CSV format.
4. Save the downloaded CSV file as "covid_data.csv" in the same repository.
5. Open the extracted repository directory and locate the "final_report.ipynb" notebook.
6. Launch Jupyter Notebook or JupyterLab on your computer.
7. Navigate to the extracted repository directory within the Jupyter interface.
8. Click on the "final_report.ipynb" notebook to open it.
9. Execute the notebook cells to reproduce the analysis using the downloaded dataset.
10. Modify the notebook as needed, such as adjusting parameters or adding additional visualizations.
    Save the modified notebook for future reference or sharing.

# Executive Summary:
1. The analysis reveals a higher mortality rate among COVID-19 positive patients, but a significant number of individuals have survived.
2. Positive COVID-19 cases show a significant correlation with underlying health conditions, as indicated by the Chi-Square test.
3. Hospitalization is negatively correlated with positive COVID-19 cases, although a substantial number of positive cases still require hospitalization.
4. Non-tobacco users have a higher proportion of positive COVID-19 cases compared to tobacco users, supported by the Chi-Square test.
5. Heart problems may not have a strong correlation with positive COVID-19 cases, as individuals without heart problems have a higher rate of positive cases.
6. Adults between the ages of 30 and 60+ are more likely to test positive for COVID-19, with a higher exposure risk for individuals in their 30s to 50s, potentially due to work and increased societal engagement.
7. In this second iteration, we explore the integration of clustering techniques into our modeling process to uncover patterns and distinct groups within the data, enhancing accuracy and interpretability.
8. Our models outperform the baseline accuracy of 48%, with the random forest model with a maximum depth of 2 and 10 estimators achieving the highest accuracy of 74.2%.
9. The implementation of this model enables confident prediction of individuals with a medium level of risk before or after COVID-19 testing, contributing to effective risk assessment and management strategies.
10. The test data achieved an accuracy of 74.2%, matching the accuracy of the validation data and surpassing the baseline by 26.2%.
 
 # Recommendations and Takeaways:
 1. Our recommendation is to utilize the current model to predict the risk level of patients presenting symptoms and with pre-existing medical conditions, considering the specific features used in this model.
2. Accurately predicting the risk level of patients enables better preparation and assessment of hospitals and clinics, ensuring the provision of necessary tools, equipment, and staff for future COVID-19 patients, regardless of their test results.
3. Since this dataset represents a specific demographic from Mexico, we aim to gather data from other countries to develop a more universal model applicable to hospitals and clinics worldwide.
4. Given more time, further research would involve acquiring more recent data, adopting different data preparation and cleaning approaches, and obtaining additional patient information such as socio-economic status and race to enhance the model's comprehensiveness. 
