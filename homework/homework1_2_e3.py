#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：My Machine Learning
@File ：homework1_2_e3.py
@Author ：xiao zhang
@Date ：2021/9/15 18:54 
'''

from sklearn import datasets, linear_model

# the data in table in exercise3
X = [6, -3, 5, 7, 6]
Y = [2, -2, -6, 3, 4]
Z = [3, 4, 2, -2, -8]
T = [28, 31, 53, 1, 35]

# define the regression model
regr = linear_model.LinearRegression()
X_train = [[X[i], Y[i], Z[i]] for i in range(len(X))]
Y_train = T

# training
regr.fit(X_train, Y_train)

[theta1, theta2, theta3] = regr.coef_
theta4 = regr.intercept_
print(theta1, theta2, theta3, theta4)
