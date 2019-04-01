import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 
class Score():
    def __init__(self, x=[], y=[], x_test=[], y_test=[]):
        self.regressor = LinearRegression()
        self.regressor.fit(x, y)
        self.model_score = [r2_score(y, self.regressor.predict(x)), r2_score(y_test, self.regressor.predict(x_test))]
        self.scored = True


    def r2_score_test_data(self, x_test=[], y_test=[]):
        return r2_score(y_test, self.regressor.predict(x_test))
 
