"""
Created on Tue Mar 12 23:14:20 2019

@author: leo
"""
from json import dumps
from numpy import r_
from models.MLinearRegression import MLR

info = {
    'title': 'Multivariate Analysis (GPD/Capita) in Europe', 
    'x': 'Measured', 
    'y': 'Predicted', 
    'path': 'data/AVG_GDP_Europe.csv'
}

# create the MLR object
multivariate_analysis = MLR(dumps(info))
print(multivariate_analysis.info.data.iloc[:, r_[1:1, 1:2]])
multivariate_analysis.set_dimensions(multivariate_analysis.info.data.iloc[:, ], multivariate_analysis.info.data.iloc[:, r_[1:1, 1:2]])
# original_data = multivariate_analysis.info.data

# central and eastern
# print(original_data.iloc[:, 1:1:2])

# multivariate_analysis.data_fetch()
# print("General Summary")
# print("-----------------------------------------------")
# print(multivariate_analysis.get_summary(0))
# print()
print("R Style Summary")
print("-----------------------------------------------")
print(multivariate_analysis.get_summary(1))
# print()
# print("Console Summary")
# print("-----------------------------------------------")
# print(multivariate_analysis.get_summary(2))
print("-----------------------------------------------")

# apply the sampling technique
multivariate_analysis.apply_cross_val()

# fit the data
multivariate_analysis.fit()
print()
multivariate_analysis.get_regression_coef()

# predict
multivariate_analysis.get_predictions()

# print(multivariate_analysis.predictions)
# print(multivariate_analysis.x_test.__sizeof__())

# plot the results
multivariate_analysis.plot() 