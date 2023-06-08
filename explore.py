# IMPORTS

# Libraries working for arrays and dataframe
import numpy as np
import pandas as pd
# Used for Data Visualization and statistical tool
from scipy import stats
from pydataset import data
import seaborn as sns
import matplotlib.pyplot as plt 

#personal files/functions
import wrangle as w



# FUNCTIONS:

def create_pie_chart(df, column_name,title):
    """
    Create a pie chart based on the specified column in the dataframe.

    Parameters:
        df (DataFrame): The dataframe containing the data.
        column_name (str): The name of the column to create the pie chart.
        title (str): The title of the pie chart.

    Returns:
        None
    """
    values = df[column_name].value_counts()
    labels = values.index.tolist()
    sizes = values.tolist()
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title(title)
    plt.show()


def plot_correlations(df):
    """
    Plot the correlations between feature variables and a target variable.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.

    Returns:
        None (displays the correlation bar chart)

    """
    target_var = 'covid_pos'
    feat_vars = ['med_level', 'gender', 'is_hospitalized', 'is_dead', 'ventilator',
       'pneumonia', 'age', 'pregnant', 'diabetes', 'copd', 'asthma',
       'immunosup', 'hypertension', 'heart_problems', 'obese', 'renal_disease',
       'smoker', 'icu', 'age_risk','risk_category']

    # Calculate correlations between feature variables and target variable
    correlations = df[feat_vars].corrwith(df[target_var]).sort_values()

    # Create a bar chart to visualize correlations
    plt.figure(figsize=(10, 6))
    plt.barh(correlations.index, correlations.values)
    plt.xlabel('Correlation with Target Variable')
    plt.title('Feature Variable Correlations with Target Variable')
    plt.show()

    
    
def create_barplot(data, x, y, title, xtick_labels=None, y_legend_label=None):
    """
    Create a bar plot using the given data.

    Args:
        data (pandas.DataFrame): The data to be plotted.
        x (str): The column name for the x-axis.
        y (str): The column name for the y-axis.
        title (str): The title of the plot.
        xtick_labels (list, optional): Custom labels for x-axis ticks. Defaults to None.
        y_legend_label (str, optional): Label for the y-axis legend. Defaults to None.
    """
    # Convert y column to numeric data type
    data[y] = pd.to_numeric(data[y])
    
    # Create bar plot
    sns.barplot(data=data, x=x, y=y)
    plt.title(title)
    
    # Plot average line
    plt.axhline(data[y].mean(), color='red', linestyle='--', label='Average')
    plt.legend()
    
    # Set x-axis tick labels if provided
    if xtick_labels is not None:
        plt.xticks(ticks=range(len(xtick_labels)), labels=xtick_labels)
    
    # Set y-axis legend label if provided
    if y_legend_label is not None:
        plt.ylabel(y_legend_label)


def countplot(df, xvariable, hvariable, title=None):
    """
    Create a countplot with tick labels for x-variable and hue-variable.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        xvariable (str): The column name of the x-variable.
        hvariable (str): The column name of the hue-variable.
        title (str, optional): Title for the countplot.

    Returns:
        None
    """
    # Create the countplot
    sns.countplot(x=xvariable, hue=hvariable, data=df)

    # Set the tick labels
    plt.xticks(ticks=[0, 1], labels=['Negative', 'Positive'])

    # Set the hue tick labels
    plt.legend(title=hvariable, labels=['No', 'Yes'])

    # Calculate the average
    average = df[hvariable].mean()


    # Set the title
    if title:
        plt.title(title)

    # Display the plot
    plt.show()
    
def chi_square_test(observed, alpha=0.05):
    "this function will calculate the chi square and print out results"
    chi2, p, dof, expected = stats.chi2_contingency(observed)
    print("Observed Contingency Table:")
    print(observed)
    print("Expected Contingency Table:")
    print(expected)
    print("Chi-Square Test Statistic:")
    print(chi2)
    print("p-value:")
    print(p)
    if p < alpha:
        print('We reject the null hypothesis.')
    else:
        print('We fail to reject the null hypothesis.')

def con_stats_test(df, categorical_variable, continuous_variable, alpha=0.05):
    """
    Perform statistical tests between a categorical variable and a continuous variable.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        categorical_variable (str): The column name of the categorical variable.
        continuous_variable (str): The column name of the continuous variable.
        alpha (float, optional): The significance level for the test. Default is 0.05.

    Returns:
        test_statistic (float): The test statistic value.
        p_value (float): The p-value associated with the test.
    """
    # Separate the continuous variable by the categories in the categorical variable
    groups = []
    for category in df[categorical_variable].unique():
        group = df[df[categorical_variable] == category][continuous_variable]
        groups.append(group)

    # Perform the appropriate statistical test based on the number of groups
    if len(groups) == 2:  # Perform independent t-test
        test_statistic, p_value = stats.ttest_ind(groups[0], groups[1])
    else:  # Perform one-way ANOVA
        test_statistic, p_value = stats.f_oneway(*groups)

    # Print the result based on the p-value and alpha
    if p_value < alpha:
        print('We reject the null hypothesis.')
    else:
        print('We fail to reject the null hypothesis.')

    return test_statistic, p_value

def cat_cont_plot(df, xvariable, yvariable, title=None):
    """
    Create a plot for a categorical variable on the x-axis and a continuous variable on the y-axis.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        xvariable (str): The column name of the categorical variable.
        yvariable (str): The column name of the continuous variable.
        title (str, optional): Title for the plot.

    Returns:
        None
    """
    # Create the plot
    sns.boxplot(x=xvariable, y=yvariable, data=df)

    # Set x-axis tick labels
    x_labels = df[xvariable].unique()
    plt.xticks(range(len(x_labels)), x_labels)

    # Set labels
    plt.xlabel(xvariable)
    plt.ylabel(yvariable)

    # Set the title
    if title:
        plt.title(title)

    # Display the plot
    plt.show()
    


def cat_cont_t_test(data, categorical_column, continuous_column, alpha=0.05):
    """
    Perform a statistical test between a categorical column and a continuous column.

    Args:
        data (pandas.DataFrame): The DataFrame containing the data.
        categorical_column (str): The column name of the categorical variable.
        continuous_column (str): The column name of the continuous variable.

    Returns:
        tuple: A tuple containing the test statistic and the p-value.

    """
    category_groups = data.groupby(categorical_column)
    category_values = data[categorical_column].unique()
    category_data = [category_groups.get_group(category)[continuous_column] for category in category_values]

    # Perform statistical test (Independent samples t-test)
    test_statistic, p_value = stats.ttest_ind(*category_data)

   # Print the result based on the p-value and alpha
    if p_value < alpha:
        print('We reject the null hypothesis.')
    else:
        print('We fail to reject the null hypothesis.')

    return test_statistic, p_value
