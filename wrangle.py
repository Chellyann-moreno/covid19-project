import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def covid_data():
    """
    Preprocesses COVID-19 data in the given DataFrame.
     Parameters:
    - df (pandas.DataFrame): Input DataFrame containing COVID-19 data.

    Returns:
    - pandas.DataFrame: Preprocessed DataFrame.
    """
    # Step 1: Retrieve data
    df=pd.read_csv('covid_data.csv')
    # Step 2: Lowercase column names
    df.rename(columns=lambda x: x.lower(), inplace=True)

    # Step 3: Rename specific columns
    df.rename(columns={'clasiffication_final': 'covid_pos',
                       'sex': 'gender',
                       'usmer': 'med_level',
                       'cardiovascular': 'heart_problems',
                       'intubed': 'ventilator',
                       'renal_chronic': 'renal_disease',
                       'tobacco': 'smoker',
                       'patient_type': 'is_hospitalized',
                       'inmsupr': 'immunosup',
                       'hipertension': 'hypertension',
                       'date_died': 'is_dead',
                       'obesity': 'obese'}, inplace=True)

    # Step 4: Drop unnecessary columns
    df.drop(columns=['medical_unit', 'other_disease'], inplace=True)

    # Step 5: Replace values 97 and 99 with NaN
    df.replace([97, 99], np.nan, inplace=True)

    # Step 6: Map values for specific columns
    df['covid_pos'] = df['covid_pos'].map({2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0})
    df['gender'] = df['gender'].map({1: 1, 2: 0})
    df.is_dead=np.where(df['is_dead'] == '9999-99-99', 0, 1)
    df['is_hospitalized'] = df['is_hospitalized'].map({1: 1, 2: 0})
    df['ventilator'] = df['ventilator'].map({1: 1, 2: 0})
    df['pneumonia'] = df['pneumonia'].map({1: 1, 2: 0})
    df['pregnant'] = df['pregnant'].map({1: 1, 2: 0})
    df['diabetes'] = df['diabetes'].map({1: 1, 2: 0})
    df['copd'] = df['copd'].map({1: 1, 2: 0})
    df['asthma'] = df['asthma'].map({1: 1, 2: 0})
    df['immunosup'] = df['immunosup'].map({1: 1, 2: 0})
    df['hypertension'] = df['hypertension'].map({1: 1, 2: 0})
    df['heart_problems'] = df['heart_problems'].map({1: 1, 2: 0})
    df['obese'] = df['obese'].map({1: 1, 2: 0})
    df['renal_disease'] = df['renal_disease'].map({1: 1, 2: 0})
    df['smoker'] = df['smoker'].map({1: 1, 2: 0})
    df['icu'] = df['icu'].map({1: 1, 2: 0})

    # Step 7: Create 'age_risk' column based on age values
    df['age_risk'] = np.where(df['age'] >= 60, 'high', 'low')

    # Step 8:Define the risk categories (low, medium, high) using sum of related illness
    #  Define the column names of the risk-related columns
    risk_columns = ['is_hospitalized', 'ventilator',
       'pneumonia', 'pregnant', 'diabetes', 'copd', 'asthma',
       'immunosup', 'hypertension', 'heart_problems', 'obese', 'renal_disease',
       'smoker', 'covid_pos', 'icu', 'age_risk']

# Calculate the sum of risk-related columns
    df['risk_sum'] = df[risk_columns].sum(axis=1)

# Define the conditions for categorizing the risks
    conditions = [
    (df['risk_sum'] <= 1),      # Low risk
    (df['risk_sum'] <= 3),      # Medium risk
    (df['risk_sum'] > 3)        # High risk
]

# Define the risk categories
    risk_categories = ['Low', 'Medium', 'High']

# Create the new 'risk_category' column based on the conditions and categories and fill dataframe nulls with 0
    df['risk_category'] = np.select(conditions, risk_categories, default='Unknown')
    df=df.fillna(0)

    return df




def split_data(df,variable):
    """This function helps divide the data into train, validate, and testing"
    """
    train, test = train_test_split(df,
                                   random_state=123, test_size=.20, stratify= df[variable])
    train, validate = train_test_split(train, random_state=123, test_size=.25, stratify= train[variable])
    return train, validate, test