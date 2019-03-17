"""
Created on Tue Mar 12 23:14:20 2019

@author: leo
"""
from json import dumps
from regression.MLinearRegression import MLR
from matplotlib.colors import ListedColormap

info = {
    'title': 'Multivariate Analysis of Income per (GPD/Capita) in Europe', 
    'x': '', 
    'y': '', 
    'file': 'data/Central and Eastern.csv'
}

# Variables
sample_split = 0.25 # sampling split -> 25% Test | 75% train
split_random_state = 0

# create the MLR object
multivariate_analysis = MLR(dumps(info))

# ==================================================================================

# summary before regression
summary = multivariate_analysis.get_summary(1)

# apply the sampling technique
multivariate_analysis.apply_cross_val(sample_split, split_random_state)

# fit the data
multivariate_analysis.fit()

# compare the original train to the predicted results
multivariate_analysis.actual_vs_prediction()

# print coefficients
# multivariate_analysis.print_coef_and_incetercept()
print("")
print("")
# plot the results
multivariate_analysis.plot(colmap=ListedColormap(['red', 'green']))

# ==================================================================================

# change the x dimension to Education
multivariate_analysis.set_dimensions(x="Education", y="Life Expectancy")

# summary before regression
summary = multivariate_analysis.get_summary(1)

# apply the sampling technique
multivariate_analysis.apply_cross_val(sample_split, split_random_state)

# fit the data
multivariate_analysis.fit()

# compare the original train to the predicted results
multivariate_analysis.actual_vs_prediction()

# print coefficients
# multivariate_analysis.print_coef_and_incetercept()
print("")
print("")
# plot the results
multivariate_analysis.plot(colmap=ListedColormap(['red','green']))