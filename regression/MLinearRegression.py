"""
Created on Mon Mar 11 19:29:35 2019
@Module: Data Analytics
@Type  : Coursework
@author: leo
"""

# generic import
import matplotlib.pyplot as plt

# targetted imports
from json import loads
from pandas import DataFrame
from structures.Info import Gatherer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class MLR():
    def __init__(self, info="{'title': '','x': '', 'y': '', 'file': ''}"):
        #
        info = loads(info)

        #
        self.info = Gatherer()

        # model
        self.summary = None

        # regressor
        self.regressor = LinearRegression(copy_X=False, fit_intercept=True, n_jobs=1, normalize=False)

        # train data
        self.x_train = None
        self.y_train = None
        
        # test data
        self.x_test = None
        self.y_test = None
        
        # predicted model
        self.predictions = None

        # get the file path & data 
        self.info.set_file(info['file'])
        self.info.set_data()
        
        # boolean conditions
        self.isFit = False
        self.isCrossValidated = False
        
        #
        self.x_dim = None  
        self.y_dim = None 

        # get the labels for Multivariate LR and the plot title
        self.info.set_plot_title(info['title'])
        self.info.set_x_label(info['x'])
        self.info.set_y_label(info['y'])
      
    def set_dimensions(self, x="", y=""):
        self.x_dim = self.info.data[[x]]
        self.y_dim = self.info.data[[y]]
        self.isFit = False
         
    def add_file(self, path=""):
        self.info.set_file(path)
    
    # applies cross-validation sampling only (for now)
    def apply_cross_val(self, test_total_size=0, random_state=0):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_dim, self.y_dim, test_size=test_total_size, random_state=random_state, shuffle=False)

    def fit(self):
        if self.isCrossValidated:
            self.regressor.fit(self.x_train, self.y_train)
        else:
            self.regressor.fit(self.x_dim, self.y_dim)
    
    def get_predictions(self):
        self.predictions = self.regressor.predict(self.x_test)
        return self.predictions

    def actual_vs_prediction(self, predict_again=False):
        if predict_again:
            self.get_predictions()
        
        if self.predictions == None:
            self.predictions = self.regressor.predict(self.x_train)

        if self.isCrossValidated:
            print("Original: ")
            print("-----------------------------------------------")
            print(self.x_train)
            print("-----------------------------------------------")
            print("Predictions: ")
            print("-----------------------------------------------")
            print(self.predictions)
            print("-----------------------------------------------")
        else:
            print("Original: ")
            print("-----------------------------------------------")
            print(self.x_dim)
            print("-----------------------------------------------")
            print("Predictions: ")
            print("-----------------------------------------------")
            print(self.predictions)
            print("-----------------------------------------------")

    # def print_coef_and_incetercept(self):
    #     if self.isFit:
    #         self.regressor.coef_
    #         for index, col_name in enumerate(self.info.):
    #             print("The coefficient for {} is {}, intercept for {} is {}".format(self.columns[index], self.regressor.coef_[index][0], self.columns[index], self.regressor.intercept_[index]))
    #     else:
    #         self.fit()
    #         self.print_coef_and_incetercept()

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
    
        # plots the dataset 
    def plot(self, x_label="", y_label="", plot_type="scatter", column="", colmap=None):
        if x_label != "" and y_label != "":
            plt.xlabel(x_label)
            plt.ylabel(y_label)
        else:
            plt.xlabel(self.info.xLabel)
            plt.ylabel(self.info.yLabel)
        
        if plot_type == 'bar':
            plt.bar(self.regressor.coef_, self.regressor.intercept_, cmap=colmap, align='center')
            plt.plot(self.y_dim.values, self.regressor.intercept_)
        
        elif plot_type == 'scatter':
            plt.scatter(self.x_dim, self.y_dim)
            plt.plot([val[0] for val in self.regressor.coef_], self.regressor.intercept_, cmap=colmap)  #[self.regressor.coef_.min(), self.regressor.coef_.max()], [self.regressor.intercept_.min(), self.regressor.intercept_.max()])
        
        else:
            plt.plot(self.info.data, self.regressor.intercept_, cmap=colmap)

        plt.title(self.info.title, loc='center')
        plt.show()
        
            