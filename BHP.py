#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.datasets import fetch_openml
import streamlit as st

# In[2]:


house_price_dataset = pd.read_csv("boston.csv")


# In[4]:


house_price_dataset


# In[73]:


print(house_price_dataset)


# In[74]:


house_price_dataframe = pd.DataFrame(boston.data, columns=boston.feature_names)


# In[75]:


house_price_dataframe.head()


# In[77]:


house_price_dataframe['price'] = boston.target


# In[78]:


house_price_dataframe.head()


# In[79]:


house_price_dataframe.shape


# In[80]:


house_price_dataframe.isnull().sum()


# In[81]:


house_price_dataframe.describe()


# In[121]:


correlation = house_price_dataframe.corr()


# In[83]:


plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')


# In[120]:


X = house_price_dataframe.drop(['price'], axis=1)
Y = house_price_dataframe['price']


# In[122]:


print(X)
print(Y)


# In[106]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)


# In[117]:


print(X.shape, X_train.shape, X_test.shape)


# In[134]:


model = XGBRegressor()


# In[135]:


model.fit(X_train, Y_train)


# In[136]:


training_data_prediction = model.predict(X_train)


# In[137]:


print(training_data_prediction)


# In[138]:


# R squared error
score_1 = metrics.r2_score(Y_train, training_data_prediction)

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)

print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)


# In[142]:


plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Predicted Price")
plt.show()


# In[140]:


test_data_prediction = model.predict(X_test)


# In[141]:


score_1 = metrics.r2_score(Y_test, test_data_prediction)

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)

print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)

