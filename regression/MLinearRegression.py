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
from numpy import array2string
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

        # 
        self.x_tran = None
        self.y_train = None
        
        #
        self.x_test = None
        self.y_test = None
        
        #
        self.predictions = None
        
        # model
        self.model = None
        
        #
        self.summary = None

        # regressor
        self.regressor = LinearRegression(copy_X=False, fit_intercept=True, n_jobs=1, normalize=False)
        
        # get the file path & data 
        self.info.set_file_path(info['file'])
        self.info.set_data()
        
        # boolean conditions
        self.isFit = False
        self.isCrossValidated = False
        self.predicted = False
        
        #
        self.x_dim = self.info.data[info['columns']]
        self.y_dim = self.info.data[info['target']]
                
        # get the labels for Multivariate LR and the plot title
        self.info.set_plot_title(info['title'])
        # self.info.set_x_label(info[''])
        self.info.set_y_label(info['target'])

        # print(self.x_dim.shape)
        # print(self.y_dim.values.reshape(1, 1))
        # slope and intercept

         
    def add_file(self, path=""):
        self.info.set_file_path(path)
    
    # applies cross-validation sampling only (for now)
    def apply_cross_val(self, test_total_size=0, random_state=0):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_dim, self.x_dim, test_size=test_total_size, random_state=random_state, shuffle=False)

    def fit(self, x="", y=""):
        if self.isCrossValidated:
            self.model = self.regressor.fit(self.x_train[[x]], self.y_train)
            self.isFit = True
        else:
            self.model = self.regressor.fit(self.x_dim[[x]], self.y_dim)
            self.isFit = True
    
    def get_predictions(self, x="", y=""):
        if self.isFit:
            if self.isCrossValidated:
                self.predictions = DataFrame(self.model.predict(self.x_test[[x]]))
                self.predicted = True
            else:
                self.predictions = DataFrame(self.model.predict(self.x_dim[[x]]))
                self.predicted = True
        else:
            self.fit()
            self.get_predictions()
            
        return self.predictions

    def actual_vs_prediction(self, predict_again=False):
        if predict_again:
            self.get_predictions()
        
        if self.isFit and not self.predicted:
            if self.isCrossValidated:
                self.predictions = DataFrame(self.model.predict(self.x_train))
            else:
                self.predictions = DataFrame(self.model.predict(self.x_dim))

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

    def print_coef_and_incetercept(self, column=""):
        print("Coefficients: ")
        print(array2string(self.model.coef_))
        print("Intercept: ")
        print(array2string(self.model.intercept_))
        #if self.isFit:
        #    data = self.x_dim
        #if self.isCrossValidated:
        #    data = self.x_train
        
            #for index, col_name in enumerate(data):
             #       print("The coefficient for {} is {}, \n intercept for {} is {}".format(col_name, self.regressor.coef_[index][0], col_name, self.regressor.intercept_[index]))
             
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
    def plot(self, x_label="", y_label="", plot_type="scatter", x="", y="", column="", colmap=None):
        if x_label != "" and y_label != "":
            plt.xlabel(x_label)
            plt.ylabel(y_label)
        else:
            self.info.set_x_label(x)
            plt.xlabel(self.info.xLabel)
            plt.ylabel(self.info.yLabel)
        
        if plot_type == 'bar':
            plt.bar(self.x_dim[[x]], self.y_dim, cmap=colmap, align='center')
            plt.plot(self.x_dim[[x]], self.predictions)
        
        elif plot_type == 'scatter':   
            plt.scatter(self.x_dim[[x]], self.y_dim, cmap=colmap)
            plt.plot(self.x_dim[[x]], self.predictions, linewidth=1)  
            
        else:
            plt.plot(self.x_dim[[x]], self.y_dim)
            plt.plot(self.x_dim[[x]], self.predictions, linewidth=1)

        plt.title(self.info.title, loc='center')
        plt.show()
        
            