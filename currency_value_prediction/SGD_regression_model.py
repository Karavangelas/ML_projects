import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load
from sklearn.base import BaseEstimator, TransformerMixin

# class used to drop the date column

class drop_column(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.drop([self.dtype], axis=1)

# load the data and separate X and y

data = pd.read_pickle('dataset.pkl')
dfX = pd.DataFrame(data['X'])
dfy = pd.DataFrame(data['y'])

# create the transformation pipeline and then apply it to the X data

pipeline = Pipeline([
    ('drop_column', drop_column("date")),
    ('imputer', SimpleImputer(strategy='mean')),
    ('std_scaler', StandardScaler())
])

dfX_set = pipeline.fit_transform(dfX)

# create joblib for pipeline

dump(pipeline, 'pipeline2.joblib')

# split the data in training and test sets

train_set, test_set = train_test_split(dfX_set, random_state=0)
train_set_y, test_set_y = train_test_split(dfy, random_state=0)

# convert y into a flattened array, as SGD regressor requires

train_set_y = np.ravel(train_set_y)

# train a model using the training data, in this case SGD regressor is selected

SGD_reg = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
SGD_reg.fit(train_set, train_set_y)

# make predictions using the test set

y_train_pred = SGD_reg.predict(train_set)
y_test_pred = SGD_reg.predict(test_set)

# create joblib for model

dump(SGD_reg, 'model2.joblib')
