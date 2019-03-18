"""
Created on Tue Mar 12 23:14:20 2019

@author: leo
"""
from json import dumps
from regression.MLinearRegression import MLR

info = {
    'title': 'Multivariate Analysis of Southern Europe', 
    'target': 'Life Expectancy',
    'file': 'data/Southern.csv',
    'columns': ['Income','Life Expectancy', 'Education']
}

# Variables
sample_split = 0.25 # sampling split -> 25% Test | 75% train
split_random_state = 0
column = 'Income'

# create the MLR object
multivariate_analysis = MLR(dumps(info))

# ==================================================================================

# summary before regression
summary = multivariate_analysis.get_summary(1)

# apply the sampling technique
# multivariate_analysis.apply_cross_val(sample_split, split_random_state)

# fit the data
multivariate_analysis.fit(x=column)

# predictions
multivariate_analysis.get_predictions(x=column)

# compare the original train to the predicted results
# multivariate_analysis.actual_vs_prediction()

print("")
print("")

# plots the results
multivariate_analysis.plot(x=column, plot_type='scatter')

# print coefficients
multivariate_analysis.print_coef_and_incetercept()

# ==================================================================================

column = 'Education'

# fit the data
multivariate_analysis.fit(x=column)

# predictions
multivariate_analysis.get_predictions(x=column)

print("")
print("")

multivariate_analysis.plot(x=column, plot_type='scatter')

# print coefficients
multivariate_analysis.print_coef_and_incetercept()
