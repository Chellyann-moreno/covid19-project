#IMPORTS
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
import explore as e

# For modeling,scaling, and clustering purposes:
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler



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
    
def run_decision_tree(X_train, X_test, y_train, y_test, max_depth):
    """
    Trains a Decision Tree classifier on the provided training data and predicts the target variable
    for the test data.

    Parameters:
        X_train (array-like): Training features, shape (n_samples, n_features).
        X_test (array-like): Test features, shape (n_samples, n_features).
        y_train (array-like): Training target variable, shape (n_samples,).
        y_test (array-like): Test target variable, shape (n_samples,).
        max_depth (int): Maximum depth of the decision tree.

    Returns:
        float: Accuracy score of the decision tree classifier on the training data.
        float: Accuracy score of the decision tree classifier on the test data.
    """
    # Create a Decision Tree classifier
    dt = DecisionTreeClassifier(random_state=123, max_depth=max_depth)
    
    # Train the classifier on the training data
    dt.fit(X_train, y_train)
    
    # Predict the target variable for the training and test data
    y_train_pred = dt.predict(X_train)
    y_test_pred = dt.predict(X_test)
    
    # Calculate and return the accuracy scores
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Decision Tree Results:")
    print(f"Train score: {train_accuracy}")
    print(f"Validate score:{test_accuracy}")
    



def run_random_forest(X_train, X_test, y_train, y_test, max_depth, n_estimators):
    """
    Trains a Random Forest classifier on the provided training data and predicts the target variable
    for the test data.

    Parameters:
        X_train (array-like): Training features, shape (n_samples, n_features).
        X_test (array-like): Test features, shape (n_samples, n_features).
        y_train (array-like): Training target variable, shape (n_samples,).
        y_test (array-like): Test target variable, shape (n_samples,).
        max_depth (int): Maximum depth of the decision trees in the random forest.
        n_estimators (int): Number of decision trees in the random forest.

    Returns:
        float: Accuracy score of the random forest classifier on the training data.
        float: Accuracy score of the random forest classifier on the test data.
    """
    # Create a Random Forest classifier
    rf = RandomForestClassifier(random_state=123, max_depth=max_depth, n_estimators=n_estimators)
    
    # Train the classifier on the training data
    rf.fit(X_train, y_train)
    
    # Predict the target variable for the training and test data
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    
    # Calculate and return the accuracy scores
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Random Forest Results:")
    print(f"Train score: {train_accuracy}")
    print(f"Validate score:{test_accuracy}")
    
def calculate_kmeans(data, n_clusters):
    """
    Perform K-means clustering on the given data.

    Parameters:
    - data (array-like): The input data to be clustered.
    - n_clusters (int): The number of clusters to create.

    Returns:
    - df (pandas.DataFrame): A DataFrame containing the original data with an additional 'cluster' column
                             indicating the cluster labels assigned by K-means.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    
    # Create a new DataFrame with the cluster labels
    df = pd.DataFrame(data)
    df['cluster'] = kmeans.labels_
    
    return df

def elbow_method(data, max_k):
    """
    Applies the elbow method to determine the optimal number of clusters (k) for KMeans clustering.

    Args:
        data (array-like): The input data to be clustered.
        max_k (int): The maximum number of clusters to consider.

    Returns:
        None

    The function calculates the within-cluster sum of squares (WCSS) for different values of k and
    plots the WCSS values against the number of clusters. The 'elbow' point in the plot is often
    considered as the optimal value of k, indicating the number of clusters where adding more clusters
    does not significantly decrease the WCSS.

    Example usage:
        data = [[1, 2], [3, 4], [5, 6], ...]  # Input data
        max_k = 10  # Maximum number of clusters to consider
        elbow_method(data, max_k)
    """
    wcss = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    
    # Plot the WCSS values
    plt.plot(range(1, max_k+1), wcss)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.title('Elbow Method')
    plt.show()
    

def add_cluster_col(scaled_df, cluster_cols):
    """
    Adds a cluster column to the scaled dataframe.

    Parameters:
        scaled_df (pandas.DataFrame): The scaled dataframe to which the cluster column will be added.
        cluster_cols (pandas.DataFrame): The dataframe containing the cluster information.

    Returns:
        pandas.DataFrame: The new dataframe with the added cluster column.

    Example:
        scaled_df:
           feature_1  feature_2  feature_3
        0   0.1        0.5        0.7
        1   0.3        0.2        0.9

        cluster_cols:
           cluster
        0  A
        1  B

        add_cluster_col(scaled_df, cluster_cols) returns:
           feature_1  feature_2  feature_3  cluster_B
        0   0.1        0.5        0.7        0
        1   0.3        0.2        0.9        1
    """
    cluster_col = pd.DataFrame(cluster_cols.iloc[:, -1])
    cluster_col = pd.get_dummies(cluster_col['cluster'],prefix='cluster', drop_first=True)
    new_df = pd.concat([scaled_df, cluster_col], axis=1)
    return new_df

def robust_scale_data(X_train, X_validate, X_test):
    """Scale the features using RobustScaler and return the scaled data as DataFrames."""
    # Initialize RobustScaler object
    scaler = RobustScaler()
    
    # Fit scaler object to training data
    scaler.fit(X_train)
    
    # Transform training, validation, and test data
    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled arrays back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_validate_scaled = pd.DataFrame(X_validate_scaled, columns=X_validate.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Return scaled data as DataFrames
    
    
def run_random_forest2(X_train, X_test, y_train, y_test, max_depth, n_estimators):
    """
    Trains a Random Forest classifier on the provided training data and predicts the target variable
    for the test data.

    Parameters:
        X_train (array-like): Training features, shape (n_samples, n_features).
        X_test (array-like): Test features, shape (n_samples, n_features).
        y_train (array-like): Training target variable, shape (n_samples,).
        y_test (array-like): Test target variable, shape (n_samples,).
        max_depth (int): Maximum depth of the decision trees in the random forest.
        n_estimators (int): Number of decision trees in the random forest.

    Returns:
        float: Accuracy score of the random forest classifier on the training data.
        float: Accuracy score of the random forest classifier on the test data.
    """
    # Create a Random Forest classifier
    rf = RandomForestClassifier(random_state=123, max_depth=max_depth, n_estimators=n_estimators)
    
    # Train the classifier on the training data
    rf.fit(X_train, y_train)
    
    # Predict the target variable for the training and test data
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    
    # Calculate and return the accuracy scores
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Random Forest Results:")
    print(f"Train score: {train_accuracy}")
    print(f"Test score:{test_accuracy}")
  


