# ML K-Means Mean-Shift algorithm
# For Data Analytics
# Qmul Data Analytics Module Coursework
# By Leo Mengesha

# plot libraries
import plotly.plotly as py
import plotly.graph_objs as go
import plotly

# ML libraries
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs

# Gatherer found in structures folder
from structures.MeanShift-Gatherer import KMM_Gatherer

class KMMS():
    def __init__(self, centers=[[1, 1], [-1, -1], [1, -1]], file_path=None):
        # actual data
        self.data = None

        # Gatherer
        self.gatherer = KMM_Gatherer()
        
        # seed
        self.seed = 4

        # clusters initial centroids
        self.centers = centers
        
        # x=attributes and y=labels
        self.x_dim = None
        self.y_dim = None

        # train set
        self.x_train = None
        self.y_train = None 

        # test set
        self.x_test = None
        self.y_test = None

        # bandwith
        self.__bandwidth = None
