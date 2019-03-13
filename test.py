"""
Created on Tue Mar 12 23:14:20 2019

@author: leo
"""

from json import dumps
from models.MLinearRegression import MLR

info = {
    'title': 'Multivariate Analysis (GPD/Capita) in Europe', 
    'x': 'x', 
    'y': 'y', 
    'path': 'data/AVG_GDP_Europe.csv'
}

# create the MLR object
multivariate_analysis = MLR(dumps(info))

# multivariate_analysis.data_fetch()
print("-----------------------------------------------")
print(multivariate_analysis.get_summary(0))
print(multivariate_analysis.get_summary(1))
print(multivariate_analysis.get_summary(2))
print("-----------------------------------------------")

# apply the sampling technique
multivariate_analysis.apply_cross_val()

# fit the data
multivariate_analysis.fit()

# predict
multivariate_analysis.get_predictions()

# plot the results
multivariate_analysis.plot() 