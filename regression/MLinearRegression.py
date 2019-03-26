"""
Created on Mon Mar 11 19:29:35 2019
@Module: Data Analytics
@Type  : Coursework
@author: leo
"""

# generic import
import matplotlib.pyplot as plt
import statsmodels.api as sm

# targetted imports
from json import loads
# from pandas import DataFrame
# from numpy import array2string
from structures.Info import Gatherer
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# from sklearn.linear_model import LinearRegression

class MLR():
    def __init__(self, info="{'title': '','x': '', 'y': '', 'file': ''}"):
        #
        info = loads(info)

        # info gatherer, i.e.: file path, data
        self.info = Gatherer()

        # train data to trian the ML model
        self.x_train = None
        self.y_train = None
        
        # test data
        self.x_test = None
        self.y_test = None

        # scores
        self.train_score = None
        self.test_score = None
        
        #
        self.predictions = None
        
        # model
        self.model = None
        
        # regressor
        self.regressor = None

        # get the file path & data 
        self.info.set_file_path(info['file'])
        self.info.set_data()
        
        # boolean conditionals to semi-automate model generation
        self.isFit = False
        self.isCrossValidated = False
        self.predicted = False
        
        # the x and y columns for regression
        self.x_dim = self.info.data[info['attributes']]
        self.y_dim = self.info.data[info['target']]
                            
        # get the labels for Multivariate LR and the plot title
        # plot mechanism will be added in the future
        self.info.set_plot_title(info['title'])      # sets the plot title for any generated plot 
        self.info.set_x_label(info['attributes'][0]) # sets the x label for a plot
        self.info.set_y_label(info['attributes'][1]) # sets the y label for a plot

        # column applied for one to one relationship between the target and predictor (i.e.: this column)
        self.applied_x_column = None

    """
    Parameters: string
    Description: sets the file new file path and fetches/sets new data 
    """
    def add_file(self, path=""):
        self.info.set_file_path(path)
        self.info.set_data()
    
    # applies cross-validation sampling only (for now)
    def apply_cross_val(self, test_total_size=0, random_state=0):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_dim, self.y_dim, test_size=test_total_size, random_state=random_state, shuffle=False)
        self.isCrossValidated = True

        # ols const version
        self.x_train_const = sm.add_constant(self.x_train)
        self.x_test_const = sm.add_constant(self.x_test)

    """
    Parametes: string
    Description: The function fits the model and populates the regressor, x_column_name is an optional parameter which
    tells fit whether:
        1 - if cross-validation has been applied on the data
        2 - and if the user wants to fit the model based on a one to one basis (Univariate Linear Regression)
    """
    def fit(self, x_column_name=''):
            if self.isCrossValidated:
                if x_column_name != '':
                    self.applied_x_column = x_column_name
                    self.model = sm.OLS(self.y_train, self.x_train[[x_column_name]])
                    self.regressor = self.model.fit()
                else:
                    self.model = sm.OLS(self.y_train, self.x_train_const)
                    self.regressor = self.model.fit()
                
                self.isFit = True
            else:
                if x_column_name != '':
                    self.applied_x_column = x_column_name
                    self.model = sm.OLS(self.y_dim, self.x_dim[[x_column_name]])
                    self.regressor = self.model.fit()
                else:
                    self.model = sm.OLS(self.y_dim, sm.add_constant(self.x_dim[[x_column_name]]))
                    self.regressor = self.model.fit()
                
                self.isFit = True
    
    """
    Parameters:  Void
    Description: runs predictions of the get the best fit regression line for the data in question 
    """
    def predict(self):
        if self.isFit:
            if self.isCrossValidated:
                x_dimensional_data = self.x_test_const
                if self.applied_x_column != None:
                    x_dimensional_data = self.x_test[[self.applied_x_column]]
                
                self.predictions = self.regressor.predict(x_dimensional_data)
                self.predicted = True
                # self.test_score = self.model.score(self.y_test, self.x_test)
            else:
                x_dimensional_data = self.x_dim
                if self.applied_x_column != None:
                    x_dimensional_data = x_dimensional_data[[self.applied_x_column]]

                self.predictions = self.regressor.predict(x_dimensional_data)
                self.predicted = True
                # self.test_score = self.model.score(self.y_test, sm.add_constant(self.x_dim))
        else:
            self.fit()
            self.predict()
            
        return self.predictions

    """
    Parameters: int, string         
    Description: returns different type of summaries based on the data in question
    summary range goes from 0 -> 3. The optional string value is to either print or return the summary
    """
    def get_summary(self, summary_type=0, process="print"):
        self.x_summary = self.x_dim.describe()
        self.y_summary = self.y_dim.describe()
        result = {
            # summary statistics
            0: [self.x_summary, self.y_summary],
            # R style summary
            1: [self.x_summary.transpose(), self.y_summary.transpose()],
            # console
            2: [self.x_summary.head(), self.y_summary.head()],
            # metrics
            3: self.regressor.summary()
        }[summary_type]
        
        if process=="print":
            print(result)
        else:
            return result
    
    
    # def score(self):
    #     if self.isCrossValidated:
            # self.train_score = r2_score(self.x_train_cp, self.y_train)
            # self.test_score = r2_score(self.x_test_cp, self.predictions)
        
            