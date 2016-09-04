import pandas
import sklearn
import matplotlib
import numpy
import math

csv = pandas.read_csv('housingsales.csv')
train, test = sklearn.cross_validation.train_test_split(csv, train_size = 0.8)

matplotlib.pyplot.plot(train['sqft_living'], train['price'],".", test['sqft_living'], test['price'], 'o')

lr = sklearn.linear_model.LinearRegression()
lr.fit(train['sqft_living'].reshape(-1,1), train['price'].reshape(-1,1))
print "The coefficient is ", lr.coef_
print "Mean Square Error ", numpy.mean((lr.predict(test['sqft_living'].reshape(-1,1)) - test['price'].reshape(-1,1))**2)
print "RMSE ", math.sqrt(numpy.mean((lr.predict(test['sqft_living'].reshape(-1,1)) - test['price'].reshape(-1,1))**2))
print "Variance score ", lr.score(test['sqft_living'].reshape(-1,1), test['price'].reshape(-1,1))

matplotlib.pyplot.plot(test['sqft_living'].reshape(-1,1), test['price'].reshape(-1,1),".",
                       test['sqft_living'].reshape(-1,1), lr.predict(test['sqft_living'].reshape(-1,1)), '-')

