import os

# Import library for data manipulation
import numpy as np 
import pandas as pd  

# Import library to preprocessing data
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Define the working directories
source_path = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(os.path.dirname(source_path), 'data')
raw_path = os.path.join(data_path, 'raw')

# Verification of the existence of the raw folder
if not os.path.exists(raw_path):
    raise ValueError("The raw folder doesn't exist")
# Verification of the existence of the train dataset file
if not os.path.exists(os.path.join(raw_path, 'train.csv')):
    raise ValueError("Please deposit the training dataset in the raw folder")

def calculate_rmse(estimator, X, y):
    y_pred = estimator.predict(X)
    return mean_squared_error(y, y_pred, squared=False)

class BasicPreprocessing():

    def __init__(self, display = False):
        # Define the dataset and select columns
        train = pd.read_csv(os.path.join(raw_path, 'train.csv'))
        submit =  pd.read_csv(os.path.join(raw_path, 'submit.csv'))
        # Removes columns with too many missing values
        dropped_columns = ['Alley', 'FireplaceQu', 'PoolQC', 'MiscFeature', 'Fence', 'Id']
        train.drop(columns=dropped_columns, inplace=True)
        submit.drop(columns=dropped_columns, inplace=True)
        # Removes outliers
        train.drop(train[(train['OverallQual'] > 9) & (train['SalePrice'] < 220000)].index, inplace=True)
        train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index, inplace=True)
        # Adds a feature corresponding to the total area
        train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
        submit['TotalSF'] = submit['TotalBsmtSF'] + submit['1stFlrSF'] + submit['2ndFlrSF']
        # Adds a feature corresponding to the sum of the year of construction and renovation
        train['YrBltAndRemod'] = train['YearBuilt'] + train['YearRemodAdd']
        submit['YrBltAndRemod'] = submit['YearBuilt'] + submit['YearRemodAdd']
        # Select features and stock it, in a constant
        FEATURES = train.drop(columns='SalePrice')
        TARGET = train.drop(columns=FEATURES.columns)
        # Separates the dataset between categorical and numerical values
        NUMERICAL = train[FEATURES.columns].select_dtypes(exclude='object')
        CATEGORICAL = train[FEATURES.columns].select_dtypes(include='object')
        # Split the dataset into training and test dataframe
        X_train, X_test, y_train, y_test = train_test_split(FEATURES, TARGET, test_size=.2, random_state=42)
        # Display the content of the training and test dataset
        if display:
            print(f"[INFO] Training dataset contains {X_train.shape[1]} features and {X_train.shape[0]} data")
            print(f"[INFO] Test dataset contains {X_test.shape[1]} features and {X_test.shape[0]} data")
        # Configuration of the pipeline with the addition of a regression model for validation
        numerical_pipe = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')),
            ('scaler', RobustScaler())
        ])
        categorical_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])
        preprocessing = ColumnTransformer(transformers=[
            ('num', numerical_pipe, NUMERICAL.columns),
            ('cat', categorical_pipe, CATEGORICAL.columns),
        ])
        pipeline = Pipeline([
            ('preprocessing', preprocessing),
            ('model', Ridge())
        ])
        # Fit the pipeline
        pipeline.fit(X_train, y_train)
        # Displays the R-MSE score on the training and test sets
        if display:
            print(f"[INFO] Train R-MSE with Ridge: {calculate_rmse(pipeline, X_train, y_train)}")
            print(f"[INFO] Test R-MSE with Ridge: {calculate_rmse(pipeline, X_test, y_test)}")

        return preprocessing