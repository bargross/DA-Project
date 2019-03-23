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

        #
        self.info = Gatherer()

        # 
        self.x_train = None
        self.y_train = None
        
        #
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
        
        # boolean conditions
        self.isFit = False
        self.isCrossValidated = False
        self.predicted = False
        
        #
        self.x_dim = self.info.data[info['attributes']]
        self.y_dim = self.info.data[info['target']]
        
        # metrics
        # self.metrics = None
                            
        # get the labels for Multivariate LR and the plot title
        self.info.set_plot_title(info['title'])
        self.info.set_x_label(info['attributes'][0])
        self.info.set_y_label(info['attributes'][1])

         
    def add_file(self, path=""):
        self.info.set_file_path(path)
    
    # applies cross-validation sampling only (for now)
    def apply_cross_val(self, test_total_size=0, random_state=0):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_dim, self.y_dim, test_size=test_total_size, random_state=random_state, shuffle=False)
        self.isCrossValidated = True

        # copies
        self.x_train_cp = self.x_train
        self.x_test_cp = self.x_test

        # ols const version
        self.x_train = sm.add_constant(self.x_train)
        self.x_test = sm.add_constant(self.x_test)

    def fit(self):
            if self.isCrossValidated:
                self.model = sm.OLS(self.y_train, self.x_train)
                self.regressor = self.model.fit()
                self.isFit = True
            else:
                self.model = sm.OLS(self.y_dim, sm.add_constant(self.x_dim))
                self.regressor = self.model.fit()
                self.isFit = True

        
    
    def predict(self):
        if self.isFit:
            if self.isCrossValidated:
                self.predictions = self.regressor.predict(self.x_test)
                self.predicted = True
                # self.test_score = self.model.score(self.y_test, self.x_test)
            else:
                self.predictions = self.regressor.predict(self.x_dim)
                self.predicted = True
                # self.test_score = self.model.score(self.y_test, sm.add_constant(self.x_dim))
        else:
            self.fit()
            self.predict()
            
        return self.predictions
             
    # 
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
            4: self.regressor.summary()
        }[summary_type]
        
        if process=="print":
            print(result)
        else:
            return result
    
    
    # def score(self):
    #     if self.isCrossValidated:
            # self.train_score = r2_score(self.x_train_cp, self.y_train)
            # self.test_score = r2_score(self.x_test_cp, self.predictions)
        
            