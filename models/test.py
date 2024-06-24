import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Append the parent directory to the system path
sys.path.append(parent_dir)


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
import Data
import dclassifier

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = Data.Data()
prices = tpot_data.prices[tpot_data.prices.index.to_period('M') == '2020-02']
X, y = dclassifier.classify_split_timeseries(prices['USDCHF'], 0.001)
training_features, testing_features, training_target, testing_target = \
            train_test_split(X, y, random_state=42)

imputer = SimpleImputer(strategy="median")
imputer.fit(training_features)
training_features = imputer.transform(training_features)
testing_features = imputer.transform(testing_features)

# Average CV score on the training set was: 1.0
exported_pipeline = GaussianNB()
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)