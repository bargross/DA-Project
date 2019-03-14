"""
Created on Mon Mar 11 19:29:35 2019
@Module: Data Analytics
@Type  : Coursework
@author: leo
"""

# generic import
import matplotlib.pyplot as plt

# targetted imports
from pandas import DataFrame
from json import loads
from numpy import r_
from structures.Info import Gatherer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class MLR():
    def __init__(self, info="{'title': '','x': '', 'y': '', 'path': ''}"):
        #
        info = loads(info)

        #
        self.info = Gatherer()

        # model
        self.summary = None

        # regressor
        self.regressor = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True)

        # dimensions for M-Analysis
        self.x_dim = []
        self.y_dim = []

        # train data
        self.x_train = None
        self.y_train = None
        
        # test data
        self.x_test = None
        self.y_test = None

        # predicted model
        self.predictions = None

        # get the file path & data 
        self.info.set_file_path(info['path'])
        self.info.set_data()

        #
        self.isFit = False
        
        #
        self.x_dim = self.info.data.iloc[info['cutoff']:, :1].values
        self.y_dim = self.info.data.iloc[info['cutoff']:, :1].values

        # get the labels for Multivariate LR and the plot title
        self.info.set_plot_title(info['title'])
        self.info.set_x_label(info['x'])
        self.info.set_y_label(info['y'])

    def set_dimensions(self, x, y):
        self.x_dim = x
        self.y_dim = y
        self.isFit = False
         
    def add_data_path(self, path=""):
        self.info.set_file_path(path)

    def reset_data(self):
        new_data = [
            self.info.data['Central and Eastern'], 
            self.info.data['Northern'],	
            self.info.data['Southern'],
            self.info.data['Western']
        ]
        self.info.data = DataFrame(data=new_data)

    def column_to_int(self, column_name=""):
            if column_name=='Central and Eastern':
                return 0 
            elif column_name=='Northern':
                return 1	
            elif column_name=='Southern':
                return 2
            elif column_name=='Western':
                return 3
            else:
                return None

    # plots the dataset 
    def plot(self, x_label="", y_label="", plot_type="bar", column=""):
        if x_label != "" and y_label != "":
            plt.xlabel(x_label)
            plt.ylabel(y_label)
        else:
            plt.xlabel(self.info.xLabel)
            plt.ylabel(self.info.yLabel)
        
        if plot_type == 'bar':
            plt.bar(self.x_train[self.column_to_int(column)], self.predictions[column], color='blue', align='center')
            plt.plot([self.x_train.min(), self.x_train.max()], [self.x_train.min(), self.x_train.max()])
            plt.title(self.info.title, loc='center')
            plt.show()

        elif plot_type == 'scatter':
            plt.scatter(self.x_train, self.predictions)
            plt.plot([self.x_train.min(), self.x_train.max()], [self.x_train.min(), self.x_train.max()])
            plt.title(self.info.title, loc='center')
            plt.show()
        else:
            plt.subplot(2,1,1)
            plt.plot([self.x_train.min(), self.x_train.max()], [self.x_train.min(), self.x_train.max()])
            plt.title(self.info.title, loc='center')
            plt.show()
    
    # applies cross-validation sampling only (for now)
    def apply_cross_val(self, test_total_size=0, random_state=0):
        x = self.x_dim
        y = self.y_dim
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_total_size, random_state=random_state)

    def fit(self):
        self.regressor.fit(self.x_train, self.y_train)
        self.isFit = True
    
    def get_predictions(self):
        self.predictions = self.regressor.predict(self.x_train)
        return self.predictions

    def actual_vs_prediction(self, predict_again=False):
        if predict_again:
            self.get_predictions()
        if self.predictions != None:
            print(self.x_train) 
            print(self.predictions)
        else:
            self.predictions = self.regressor.predict(self.x_train)
            print(self.x_train)
            print(self.predictions)
    
    def print_regression_coef(self):
        if self.isFit:
            train = iter(self.x_train)
            train = next(train) 
            for coeff, col_name in enumerate(DataFrame(data=train).columns):
                    print("The coefficient for {} is {}".format(col_name, self.regressor.coef_[0][coeff]))
            # return self.regressor.coef_
        else:
            self.fit()
            self.print_regression_coef()

    #
    def get_summary(self, summary_type=0):
        summary = self.info.summary()
        return {
            # summary statistics
            0: summary,
            # R style summary
            1: summary.transpose(),
            # console
            2: summary.head()
        }[summary_type]
    
    def change_data_file(self, path="", xlabel="", ylabel="", title="", cutoff=0):
        self.info.set_file_path(path)
        self.info.set_data()

        #
        self.isFit = False
        
        #
        self.x_dim = self.info.data.iloc[cutoff:, :1].values
        self.y_dim = self.info.data.iloc[cutoff:, :1].values

        # get the labels for Multivariate LR and the plot title
        self.info.set_plot_title(title)
        self.info.set_x_label(xlabel)
        self.info.set_y_label(ylabel)
        
            