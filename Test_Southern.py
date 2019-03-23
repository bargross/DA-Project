from json import dumps
from regression.MLinearRegression import MLR

info = {
    'title': 'Multivariate Analysis of Southern Europe', 
    'file': 'data/Southern.csv',
    'target': 'Life Expectancy', 
    'attributes': ['Income', 'Education']
}

# Variables
sample_split = 0.25 # sampling split -> 25% Test | 75% train
split_random_state = 0
column = 'Income'

# create the MLR object
multivariate_analysis = MLR(dumps(info))

# ==================================================================================

# summary before regression
multivariate_analysis.get_summary(summary_type=0, process="print")

# apply the sampling technique
# multivariate_analysis.apply_cross_val(sample_split, split_random_state)

# fit the data
multivariate_analysis.fit(x=info['attributes'][0], y=info['attributes'], z=info['target'], num_of_dims=3)

# predictions
multivariate_analysis.get_predictions(x=column, num_of_dims=3)

# compare the original train to the predicted results
# multivariate_analysis.actual_vs_prediction()

print("")
print("")

# plots the results
multivariate_analysis.plot(z_label=info['target'], x="Income", y="Education", plot_type='scatter3d')

# print coefficients
multivariate_analysis.print_coef_and_incetercept()