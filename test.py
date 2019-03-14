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

# Variables
sample_split = 0.20 # sampling split -> 20% Test | 80% train
split_random_state = 1

# create the MLR object
multivariate_analysis = MLR(dumps(info))

# set the x and y dimensions
multivariate_analysis.set_dimensions(multivariate_analysis.info.data.iloc[:, :-1], multivariate_analysis.info.data.iloc[:, :1])

# summary before regression
summary = multivariate_analysis.get_summary(1)


# apply the sampling technique
multivariate_analysis.apply_cross_val(sample_split, split_random_state)

# fit the data
multivariate_analysis.fit()
print(multivariate_analysis.x_train.__sizeof__())
print(multivariate_analysis.x_train.get_predictions().__sizeof__())
multivariate_analysis.get_regression_coef()

# original summary
print("Summary before M-Linear Regression (R Style Summary)")
print("-----------------------------------------------")
print(multivariate_analysis.get_summary(1))
print("-----------------------------------------------")
print()
print("Predictions: ")
print("-----------------------------------------------")
print(multivariate_analysis.get_predictions())
print("-----------------------------------------------")

# print(multivariate_analysis.predictions)
# print(multivariate_analysis.x_test.__sizeof__())

# plot the results
multivariate_analysis.plot() 