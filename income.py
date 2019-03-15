"""
Created on Tue Mar 12 23:14:20 2019

@author: leo
"""
from json import dumps
from numpy import r_
from models.MLinearRegression import MLR

info = {
    'title': 'Multivariate Analysis of Income per (GPD/Capita) in Europe', 
    'x': 'Measured', 
    'y': 'Predicted', 
    'path': 'data/AVG_GDP_Europe.csv',
    'from_cell': 75
}

# Variables
sample_split = 0.25 # sampling split -> 25% Test | 75% train
split_random_state = 1

# create the MLR object
multivariate_analysis = MLR(dumps(info))


# ==================================================================================

# set the data to the dimensions we want
# multivariate_analysis.reset_data()

# summary before regression
summary = multivariate_analysis.get_summary(1)


# apply the sampling technique
multivariate_analysis.apply_cross_val(sample_split, split_random_state)

# fit the data
multivariate_analysis.fit()

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
print("")

# print coefficients
multivariate_analysis.print_regression_coef()
print("")
print("")
# plot the results
multivariate_analysis.plot(plot_type='scatter')
# multivariate_analysis.plot(column='Northern') 
# multivariate_analysis.plot(column='Southern')
# multivariate_analysis.plot(column='Western')