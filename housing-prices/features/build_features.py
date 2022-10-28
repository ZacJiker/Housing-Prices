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

SEED = 42

class BasicPreprocessing:

    def __init__(self):
        # Define the working directories
        self.source_path = os.path.dirname(os.path.dirname(__file__))
        self.data_path = os.path.join(os.path.dirname(self.source_path), 'data')
        self.raw_path = os.path.join(self.data_path, 'raw')
        self.processed_path = os.path.join(self.data_path, 'processed')
        # Verification of the existence of the raw folder
        if not os.path.exists(self.raw_path):
            raise ValueError("The raw folder doesn't exist")
        # Verification of the existence of the train dataset file
        if not os.path.exists(os.path.join(self.raw_path, 'train.csv')):
            raise ValueError("Please deposit the training dataset in the raw folder")
        # Define the dataset and select columns
        self.train = pd.read_csv(os.path.join(self.raw_path, 'train.csv'))
        self.submit =  pd.read_csv(os.path.join(self.raw_path, 'submit.csv'))
        # Removes columns with too many missing values
        dropped_columns = ['Alley', 'FireplaceQu', 'PoolQC', 'MiscFeature', 'Fence', 'Id']
        self.train.drop(columns=dropped_columns, inplace=True)
        self.submit.drop(columns=dropped_columns, inplace=True)
        # Removes outliers
        self.train.drop(self.train[(self.train['OverallQual'] > 9) & (self.train['SalePrice'] < 220000)].index, inplace=True)
        self.train.drop(self.train[(self.train['GrLivArea'] > 4000) & (self.train['SalePrice'] < 300000)].index, inplace=True)
        # Adds a feature corresponding to the total area
        self.train['TotalSF'] = self.train['TotalBsmtSF'] + self.train['1stFlrSF'] + self.train['2ndFlrSF']
        self.submit['TotalSF'] = self.submit['TotalBsmtSF'] + self.submit['1stFlrSF'] + self.submit['2ndFlrSF']
        # Adds a feature corresponding to the sum of the year of construction and renovation
        self.train['YrBltAndRemod'] = self.train['YearBuilt'] + self.train['YearRemodAdd']
        self.submit['YrBltAndRemod'] = self.submit['YearBuilt'] + self.submit['YearRemodAdd']

    def run(self):
        # Select features and stock it, in a constant
        FEATURES = self.train.drop(columns='SalePrice').columns
        TARGET = self.train.drop(columns=FEATURES).columns
        # Separates the dataset between categorical and numerical values
        NUMERICAL = self.train[FEATURES].select_dtypes(exclude='object').columns
        CATEGORICAL = self.train[FEATURES].select_dtypes(include='object').columns
        # Separation of the data set into a training set and a test set
        X_train, X_test, y_train, y_test = train_test_split(self.train[FEATURES], self.train[TARGET],
                                                            test_size=.2, random_state=SEED)
        # Definition of data processing methods
        num_imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
        num_scaler = RobustScaler()
        cat_imputer = SimpleImputer(strategy='most_frequent')
        cat_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False)
        # Execution of the processing chain on the training data
        train_num_imputed = num_imputer.fit_transform(X_train[NUMERICAL])
        train_num_scaled = num_scaler.fit_transform(train_num_imputed)
        train_cat_imputed = cat_imputer.fit_transform(X_train[CATEGORICAL])
        train_cat_encoded = cat_encoder.fit_transform(train_cat_imputed)
        # Execution of the processing chain on the test data
        test_num_imputed = num_imputer.transform(X_test[NUMERICAL])
        test_num_scaled = num_scaler.transform(test_num_imputed)
        test_cat_imputed = cat_imputer.transform(X_test[CATEGORICAL])
        test_cat_encoded = cat_encoder.transform(test_cat_imputed)
        # Merging of data sets
        train_preprocessed = np.concatenate((train_num_scaled, train_cat_encoded), axis=1)
        test_preprocessed = np.concatenate((test_num_scaled, test_cat_encoded), axis=1)

        return train_preprocessed, test_preprocessed, y_train, y_test