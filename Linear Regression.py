
# coding: utf-8

# In[391]:

import pandas
import sklearn
import matplotlib
import numpy
import math


# In[392]:

csv = pandas.read_csv('housingsales.csv')
train, test = sklearn.cross_validation.train_test_split(csv, train_size = 0.8, random_state=0)


# In[393]:

matplotlib.pyplot.plot(train['sqft_living'], train['price'],".", test['sqft_living'], test['price'], 'o')


# In[394]:

lr = sklearn.linear_model.LinearRegression()
lr.fit(train['sqft_living'].reshape(-1,1), train['price'].reshape(-1,1))
print "The coefficient is ", lr.coef_
print "Mean Square Error ", numpy.mean((lr.predict(test['sqft_living'].reshape(-1,1)) - test['price'].reshape(-1,1))**2)
print "RMSE ", math.sqrt(numpy.mean((lr.predict(test['sqft_living'].reshape(-1,1)) - test['price'].reshape(-1,1))**2))
print "Variance score ", lr.score(test['sqft_living'].reshape(-1,1), test['price'].reshape(-1,1))


# In[395]:

matplotlib.pyplot.plot(test['sqft_living'].reshape(-1,1), test['price'].reshape(-1,1),".",
                       test['sqft_living'].reshape(-1,1), lr.predict(test['sqft_living'].reshape(-1,1)), '-')


# In[396]:

features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']

lrm = sklearn.linear_model.LinearRegression()
lrm.fit(train[features], train['price'])

print "The coefficients are ", lrm.coef_
print "Mean Square Error ", numpy.mean((lrm.predict(test[features]) - test['price'])**2)
print "RMSE ", math.sqrt(numpy.mean((lrm.predict(test[features]) - test['price'])**2))
print "Variance score ", lrm.score(test[features], test['price'])


# In[397]:

house1 = csv[csv.id == 5309101200]
print house1
print "Features Model ", lrm.predict(house1[features])
print "Sqft Model ", lr.predict(house1['sqft_living'].reshape(-1,1))


# In[398]:

house2 = csv[csv.id == 1925069082]
print house2
print "Features Model ", lrm.predict(house2[features])
print "Sqft Model ", lr.predict(house2['sqft_living'].reshape(-1,1))

