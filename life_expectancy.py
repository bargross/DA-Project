from income import multivariate_analysis,info, sample_split, split_random_state

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
# multivariate_analysis.plot(column='Central and Eastern')
# multivariate_analysis.plot(column='Northern') 
# multivariate_analysis.plot(column='Southern')
# multivariate_analysis.plot(column='Western')  