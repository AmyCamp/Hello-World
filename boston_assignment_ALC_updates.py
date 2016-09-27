
# coding: utf-8

# ## Boston Housing Assignment
# 
# In this assignment you'll be using linear regression to estimate the cost of house in boston, using a well known dataset.
# 
# Goals:
# +  Measure the performance of the model I created using $R^{2}$ and MSE
# > Learn how to use sklearn.metrics.r2_score and sklearn.metrics.mean_squared_error
# +  Implement a new model using L2 regularization
# > Use sklearn.linear_model.Ridge or sklearn.linear_model.Lasso 
# +  Get the best model you can by optimizing the regularization parameter.   

# In[1]:

from sklearn import datasets
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.linear_model import Ridge


# In[2]:

bean = datasets.load_boston()
print bean.DESCR


# In[3]:

def load_boston():
    scaler = StandardScaler()
    boston = datasets.load_boston()
    #where X are features and Y is the response vector
    X=boston.data
    y=boston.target
    X = scaler.fit_transform(X)
    return train_test_split(X,y)


# In[4]:

X_train, X_test, y_train, y_test = load_boston()


# In[5]:

X_train.shape


# ### Fitting a Linear Regression
# 
# It's as easy as instantiating a new regression object (line 1) and giving your regression object your training data
# (line 2) by calling .fit(independent variables, dependent variable)
# 
# 

# In[6]:

clf = LinearRegression()
clf.fit(X_train, y_train)


# <ul>
# <li>Looking at the train data

# In[7]:

zip (y_train, clf.predict(X_train))


# In[8]:

mse = mean_squared_error(y_train, clf.predict(X_train))
print("MSE: %f" % mse)


# In[9]:

R2 = r2_score(y_train, clf.predict(X_train))
print("R2: %f" % R2)


# In[10]:

from sklearn.linear_model import Lasso

alpha = .000001
lasso = Lasso(alpha=alpha)

y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)
print(lasso)
print("r^2 on test data : %f" % r2_score_lasso)


# ### Making a Prediction
# X_test is our holdout set of data.  We know the answer (y_test) but the computer does not.   
# 
# Using the command below, I create a tuple for each observation, where I'm combining the real value (y_test) with
# the value our regressor predicts (clf.predict(X_test))
# 
# Use a similiar format to get your r2 and mse metrics working.  Using the [scikit learn api](http://scikit-learn.org/stable/modules/model_evaluation.html) if you need help!

# In[11]:

zip (y_test, clf.predict(X_test))


# In[12]:

mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %f" % mse)


# In[13]:

import math
math.sqrt(mse)
#RMSE


# In[14]:

R2 = r2_score(y_test, clf.predict(X_test))
print("R2: %f" % R2)


# In[15]:

from sklearn.linear_model import Lasso

alpha = .000001
lasso = Lasso(alpha=alpha)

y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)
print(lasso)
print("r^2 on test data : %f" % r2_score_lasso)


# In[ ]:



