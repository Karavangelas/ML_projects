import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# load the data and separate X and y

data = pd.read_pickle('dataset.pkl')
dfX = pd.DataFrame(data['X'])
dfy = pd.DataFrame(data['y'])

# drop the data column

dfX_set = dfX.drop(['date'], axis=1)

# create the transformation pipeline and then apply it to the X data

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('std_scaler', StandardScaler())
])

dfX_set = pipeline.fit_transform(dfX_set)

#  create joblib for pipeline

dump(pipeline, 'pipelineridge.joblib')

# split the data in training and test sets

train_set, test_set = train_test_split(dfX_set, random_state=0)
train_set_y, test_set_y = train_test_split(dfy, random_state=0)

# train a model using the training data, in this case Ridge regression selected

ridge = Ridge(alpha=1.0)
ridge.fit(train_set, train_set_y)

# make predictions using the test set

y_train_pred = ridge.predict(train_set)
y_test_pred = ridge.predict(test_set)

# create joblib for model

dump(lin_reg, 'modelridge.joblib')
