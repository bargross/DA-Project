# Qmul Data Analytics Module Coursework
# By Leo Mengesha

from pandas import read_csv
from .Input import Input
# import sys

class Gatherer():
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

    def set_file_path(self, path=""):
        if path != "":
            self.__file_path = path
        else:
            self.__file_path = self.__info_gatherer.get_file_path()
            
    def set_x_label(self, xlabel=""):
        if xlabel != "":
            self.xLabel = xlabel
        else:
            self.xLabel = self.__info_gatherer.get_x_label()
    
    def set_y_label(self, ylabel=""):
        if ylabel != "":
            self.yLabel = ylabel
        else:
            self.yLabel = self.__info_gatherer.get_y_label()
    
    def set_plot_title(self, title=""):
        if title != "":
            self.title = title
        else:
            self.title = self.__info_gatherer.get_plot_title()
    
    def set_data(self):
        self.data = read_csv(self.__file_path, sep=',')

    def summary(self):
        return self.data.describe()