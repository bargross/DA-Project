import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  
# %matplotlib inline


class Input:
    #
    def get_file_path(self):
        return input('Insert The File Path')

    def get_x_label(self):
        return input('Insert X Label')
    
    def get_y_label(self):
        return input('Insert Y Label')
    
    def get_plot_title(self):
        return input('Insert Plot Title')


class Requirements():
    def __init__(self):
        self.__info_gatherer = Input()
        
        # actual data
        self.data = None

        # data-file path 
        self.__file_path = None

        # chosen x and y label for multivariate linear regression
        self.xLabel = None
        self.yLabel = None

        # plot title
        self.title = None

    def set_file_path(self):
        self.__file_path =  self.__info_gatherer.get_file_path()
    def set_x_label(self):
        self.xLabel = self.__info_gatherer.get_x_label()
    def set_y_label(self):
        self.yLabel = self.__info_gatherer.get_y_label()
    def set_plot_title(self):
        self.title = self.__info_gatherer.get_plot_title()
    def set_data(self):
        self.data = pd.read_csv(self.__file_path)

class MLR():
    def __init__(self):
        self.info = Requirements()

        # regressor
        self.regressor = LinearRegression()

        # dimensions for M-Analysis
        self.x_dim = None
        self.y_dim = None

        # train data
        self.x_train = None
        self.y_train = None
        
        # test data
        self.x_test = None
        self.y_test = None

        # predicted model
        self.predictions = None

        # get the file path & data 
        self.info.set_file_path()
        self.info.set_data()
        self.x_dim = self.info.data.iloc[:, :1].values
        self.y_dim = self.info.data.iloc[:, :1].values

        # get the labels for Multivariate LR and the plot title
        self.info.set_plot_title()
        self.info.set_x_label()
        self.info.set_y_label()

    # optional, it displays statistics of the dataset
    def show_stat_details(self):
        self.info.data.describe()

    # plots the dataset 
    def plot(self, x_label="#", y_label="#"):
        #
        xlabel = self.info.xLabel
        ylabel = self.info.yLabel

        #
        self.info.data.plot(x=xlabel, y=ylabel, style='o')
        
        #
        plt.title(self.info.title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
    
    # applies cross-validation sampling only for now
    def apply_cross_val(self, test_total_size=0.2, random_state=0):
        x = self.x_dim
        y = self.y_dim
        self.x_train, self.y_train, self.x_test, self.y_test = train_test_split(x, y, test_size=test_total_size, random_state=random_state)

    def fit(self):
        self.regressor.fit(self.x_train, self.y_train)
    
    def get_predictions(self):
        self.predictions = self.regressor.predict(self.x_test)
        return self.predictions

    def actual_vs_prediction(self, predict=False):
        if predict:
            self.get_predictions()

        if self.predictions != None: 
            print(self.predictions)
        else:
            self.predictions = self.regressor.predict(self.x_test)
            print(self.predictions)
            