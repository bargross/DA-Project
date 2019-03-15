from income import multivariate_analysis,info, sample_split, split_random_state

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
# multivariate_analysis.plot()