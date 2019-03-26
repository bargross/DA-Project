"""
Created on Tue Mar 12 23:14:20 2019

@author: leo
"""
from json import dumps
from regression.MLinearRegression import MLR

info = {
    'title': '', 
    'file': 'data/Southern.csv',
    'target': 'Life Expectancy',
    'attributes': ['Income', 'Education']
}

# Variables
sample_split = 0.25 # sampling split -> 25% Test | 75% train
split_random_state = 0

# create the MLR object
multivariate_analysis = MLR(dumps(info))

# ==================================================================================

# summary before regression
# multivariate_analysis.get_summary(summary_type=0)

# apply the sampling technique
multivariate_analysis.apply_cross_val(sample_split, split_random_state)

# fit the data
multivariate_analysis.fit()

# predictions
multivariate_analysis.predict()

# for univariate LR tests
# column = 'Income'
# column = 'Education'

# for univariate linear regression
# multivariate_analysis.fit(x_column_name=column)

# oututs a summary of the model in the cmd/terminal/etc...
multivariate_analysis.get_summary(summary_type=3)