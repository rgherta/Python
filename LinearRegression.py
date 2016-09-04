#Importing Libraries
import pandas
import sklearn
import matplotlib
import numpy
import math

#reading data and splitting into training/test
csv = pandas.read_csv('housingsales.csv')
train, test = sklearn.cross_validation.train_test_split(csv, train_size = 0.8)

#Plotting all data
matplotlib.pyplot.plot(train['sqft_living'], train['price'],".", test['sqft_living'], test['price'], 'o')

#Creating and training model
lr = sklearn.linear_model.LinearRegression()
lr.fit(train['sqft_living'].reshape(-1,1), train['price'].reshape(-1,1))

#Output model coefficients
print "The coefficient is ", lr.coef_
print "Mean Square Error ", numpy.mean((lr.predict(test['sqft_living'].reshape(-1,1)) - test['price'].reshape(-1,1))**2)
print "RMSE ", math.sqrt(numpy.mean((lr.predict(test['sqft_living'].reshape(-1,1)) - test['price'].reshape(-1,1))**2))
print "Variance score ", lr.score(test['sqft_living'].reshape(-1,1), test['price'].reshape(-1,1))

#plot predicted test
matplotlib.pyplot.plot(test['sqft_living'].reshape(-1,1), test['price'].reshape(-1,1),".",
                       test['sqft_living'].reshape(-1,1), lr.predict(test['sqft_living'].reshape(-1,1)), '-')

