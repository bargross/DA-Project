"""
Created on Tue Mar 12 23:14:20 2019

@author: leo
"""

# from importlib import reload
from json import dumps
from regression.MLinearRegression import MLR


info = {
    'title': 'Multivariate Analysis of Central and Eastern Europe', 
    'file': 'data/Central_and_Eastern.csv',
    'target': 'Life Expectancy', 
    'attributes': ['Income', 'Education']
}

# Variables
sample_split = 1/3 # sampling split -> 25% Test | 75% train
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

#
multivariate_analysis.get_summary(summary_type=4)

#
# multivariate_analysis.score()