#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.datasets import fetch_openml

# Mengambil dataset Boston dari openml
boston = fetch_openml(name='boston', version=1, as_frame=True)
house_price_dataframe = boston.frame

# Menampilkan beberapa baris pertama dari dataset
print(house_price_dataframe.head())

# Menambahkan kolom 'price' ke dataframe
house_price_dataframe['price'] = house_price_dataframe['target']

# Menampilkan bentuk dan informasi dari dataset
print(house_price_dataframe.shape)
print(house_price_dataframe.isnull().sum())
print(house_price_dataframe.describe())

# Menghitung korelasi
correlation = house_price_dataframe.corr()

# Visualisasi korelasi
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')

# Memisahkan fitur dan target
X = house_price_dataframe.drop(['price', 'target'], axis=1)
Y = house_price_dataframe['price']

print(X)
print(Y)

# Membagi data menjadi data latih dan data uji
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

# Membuat dan melatih model
model = XGBRegressor()
model.fit(X_train, Y_train)

# Prediksi pada data latih
training_data_prediction = model.predict(X_train)
print(training_data_prediction)

# Menghitung error pada data latih
score_1 = metrics.r2_score(Y_train, training_data_prediction)
score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)

print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)

# Visualisasi hasil prediksi pada data latih
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Predicted Price")
plt.show()

# Prediksi pada data uji
test_data_prediction = model.predict(X_test)

# Menghitung error pada data uji
score_1 = metrics.r2_score(Y_test, test_data_prediction)
score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)

print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)
