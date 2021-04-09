# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 20:03:23 2020

@author: vinay
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('covid_19.csv')
x=(dataset.iloc[:,:-1].values)
y=dataset.iloc[:,1].values


from sklearn.preprocessing import PolynomialFeatures
pf=PolynomialFeatures(degree=5)
x_poly=pf.fit_transform(x)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_poly,y)
c=regressor.predict(x_poly)

plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x_poly),color='blue',label='with lockdown')
plt.xlabel('days')
plt.ylabel('no of cases')
plt.show()
z=regressor.predict(pf.fit_transform(44))
a=y[42]-y[41]
b=c[42]-c[41]
print(a)
print(b)

total_cases=[]
for i in range(43,54):
    d=regressor.predict(pf.fit_transform(i))
    total_cases.append(d)
total_cases=np.array(total_cases)
plt.plot(range(43,54),total_cases,color='blue')

week_1=[]
s=x_poly[0:22,:]
r=y[0:22]
regressor.fit(s,r)
z=regressor.predict(pf.fit_transform(0))
for i in range(0,54):
    d=regressor.predict(pf.fit_transform(i))
    week_1.append(d)
week_1=np.array(week_1)
plt.plot(range(0,54),week_1,color='green',label='without lockdown')

plt.legend()

