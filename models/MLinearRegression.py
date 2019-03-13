"""
Created on Mon Mar 11 19:29:35 2019
@Module: Data Analytics
@Type  : Coursework
@author: leo
"""

import matplotlib.pyplot as plt
from json import loads

from structures.Info import Gatherer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class MLR():
    def __init__(self, info="{'title': '','x': '', 'y': '', 'path': ''}"):
        #
        info = loads(info)

        print(info)
        
        #
        self.info = Gatherer()

        # model
        self.summary = None

        # regressor
        self.regressor = LinearRegression()

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
        self.x_dim = self.info.data.iloc[:, :1].values
        self.y_dim = self.info.data.iloc[:, :1].values

        # get the labels for Multivariate LR and the plot title
        self.info.set_plot_title(info['title'])
        self.info.set_x_label(info['x'])
        self.info.set_y_label(info['y'])

    # def data_fetch(self, path=""):
    #     self.info.set_file_path(path)
    #     self.info.set_data()

    # optional, it displays the statistics of the dataset
    def show_stat_details(self):
        self.info.data.describe()
        
    def add_data_path(self, path=""):
        self.info.set_file_path(path)

    # plots the dataset 
    def plot(self, x_label="# of x", y_label="# of y"):
        #
        xlabel = None
        ylabel = None

        if x_label != "" and y_label != "":
            xlabel = x_label
            ylabel = y_label
        else:
            xlabel = self.info.xLabel
            ylabel = self.info.yLabel

        #
        self.info.data.plot(x=xlabel, y=ylabel, style='o')
        
        #
        plt.title(self.info.title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
    
    # applies cross-validation sampling only (for now)
    def apply_cross_val(self, test_total_size=0.2, random_state=0):
        x = self.x_dim
        y = self.y_dim
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_total_size, random_state=random_state)

    def fit(self):
        self.model = self.regressor.fit(self.x_train, self.y_train)
    
    def get_predictions(self):
        self.predictions = self.regressor.predict(self.x_test)
        return self.predictions

    def actual_vs_prediction(self, predict_again=False):
        if predict_again:
            self.get_predictions()

        if self.predictions != None: 
            print(self.predictions)
        else:
            self.predictions = self.regressor.predict(self.x_test)
            print(self.predictions)
    #
    #
    def get_summary(self, type=0):
        summary = self.info.summary()
        return {
            # summary statistics
            0: summary,
            # R style summary
            1: summary.transpose(),
            # console
            2: summary.head()
        }
        
            