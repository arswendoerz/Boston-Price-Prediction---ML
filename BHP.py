#!/usr/bin/env python
# coding: utf-8

# In[125]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler


# In[105]:


house_price_dataset = pd.read_csv("boston.csv")


# In[106]:


house_price_dataset


# In[107]:


print(house_price_dataset)


# In[126]:


data = fetch_openml(data_id=531)


# In[127]:


house_price_dataframe = pd.DataFrame(data=data.data, columns=data.feature_names)


# In[128]:


house_price_dataframe.head()


# In[129]:


house_price_dataframe['PRICE'] = data.target


# In[112]:


house_price_dataframe.head()


# In[130]:


house_price_dataframe.shape


# In[114]:


house_price_dataframe.isnull().sum()


# In[115]:


house_price_dataframe.describe()


# In[116]:


correlation = house_price_dataframe.corr()


# In[117]:


plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')


# In[118]:


X = house_price_dataframe.drop(['PRICE'], axis=1)
Y = house_price_dataframe['PRICE']


# In[131]:


categorical_cols = ['CHAS', 'RAD']


# In[132]:


X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)


# In[133]:


X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, test_size=0.2, random_state=2)


# In[134]:


print(X.shape, X_train.shape, X_test.shape)


# In[136]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[135]:


model = XGBRegressor()


# In[137]:


model.fit(X_train, Y_train)


# In[138]:


training_data_prediction = model.predict(X_train)


# In[139]:


print(training_data_prediction)


# In[140]:


# R squared error
score_1 = metrics.r2_score(Y_train, training_data_prediction)

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)

print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)


# In[141]:


plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Predicted Price")
plt.show()


# In[142]:


test_data_prediction = model.predict(X_test)


# In[143]:


score_1 = metrics.r2_score(Y_test, test_data_prediction)

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)

print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)

