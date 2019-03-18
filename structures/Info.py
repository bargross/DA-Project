# Qmul Data Analytics Module Coursework
# By Leo Mengesha

from pandas import read_csv
from .Input import Input

class Gatherer():
    def __init__(self):
        self.__info_gatherer = Input()
        
        # actual data
        self.data = None

        #
        self.__file = None 

        # chosen x and y label for multivariate linear regression
        self.xLabel = None
        self.yLabel = None

        # plot title
        self.title = None

    def set_file_path(self, new_file=""):
        self.__file = new_file

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
        self.data  = read_csv(self.__file, sep=',')

        # print(self.data)


    # def __get_index(self, values, name):
    #     for index, value in enumerate(values):
    #         if value == name:
    #             return index
    #         else:
    #             return -1

    # def get_dimensions(self, column_name=""):
    #     if self.dataframes.__sizeof__() >= 1:
    #         if self.files_cutoff.__sizeof__() > 0:
    #             dataframes = []
    #             files_extracted = []
    #             for cutoff in self.files_cutoff:
    #                 files_extracted.insert(cutoff['file_name'])
    #                 dataframes.insert(self.dataframes[self.__get_index(self.files, cutoff['file_name'])][cutoff['from_start']:cutoff['to_end']])
    #             for file_name in self.files:
    #                 if files_extracted.__contains__(file_name):
    #                     dataframes.insert(dataframes[self.files.index(file_name)][[column_name]])
    #             return dataframes
    #         else: 
    #             yield DataFrame([dataframe[[column_name]] for dataframe in self.dataframes])
   
    def summary(self):
        return self.data.describe()