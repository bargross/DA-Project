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
    'cutoff': 77
}

# Variables
sample_split = 0.2 # sampling split -> 20% Test | 80% train
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

# print coefficients
multivariate_analysis.print_regression_coef()

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

# plot the results
multivariate_analysis.plot(column='Central and Eastern')
multivariate_analysis.plot(column='Northern') 
multivariate_analysis.plot(column='Southern')
multivariate_analysis.plot(column='Western')  

# ==================================================================================

multivariate_analysis.change_data_file(
    path='data/AVG_Education_Europe_(mean years of schooling).csv',
    xlabel=info['x'],
    ylabel=info['y'],
    title='Multivariate Analysis of mean years of schooling in Europe'
)

# summary before regression
summary = multivariate_analysis.get_summary(1)


# apply the sampling technique
multivariate_analysis.apply_cross_val(sample_split, split_random_state)

# fit the data
multivariate_analysis.fit()

# print coefficients
multivariate_analysis.print_regression_coef()

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

# plot the results
multivariate_analysis.plot(column='Central and Eastern')
multivariate_analysis.plot(column='Northern') 
multivariate_analysis.plot(column='Southern')
multivariate_analysis.plot(column='Western')  

# ==================================================================================

multivariate_analysis.change_data_file(
    path='data/AVG_Life_expectancy_Europe.csv',
    xlabel=info['x'],
    ylabel=info['y'],
    title='Multivariate Analysis of Life Expectancy in Europe'
)

# summary before regression
summary = multivariate_analysis.get_summary(1)


# apply the sampling technique
multivariate_analysis.apply_cross_val(sample_split, split_random_state)

# fit the data
multivariate_analysis.fit()

# print coefficients
multivariate_analysis.print_regression_coef()

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

# plot the results
multivariate_analysis.plot()