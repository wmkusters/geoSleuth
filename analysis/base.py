import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import sys


class BaseAnalyzer:
    """
    Top level analyzer, used for inheritance.
    """

    def __init__(self, filename):
        """
        Prepare df for analysis (train test split)

        parameters:
            filename: location of csv file to analyze

        """

        self.result_df = pd.read_csv(filename)
        train, test = train_test_split(self.result_df, test_size=0.1, random_state=42)
        train_feature = np.array(train['feature']).reshape(-1, 1)
        # calculate num_crimes by groupby.count()
        train_crimes = train.groupby('bin_id').count()['object_id'].rename(columns={'object_id': 'num_crimes'})
        # TODO finish this part
        # self.train_feature = 
        # self.train_crimes = 
        # self.test_feature = 
        # self.test_crimes = 


    def linear_model(self, plot=False):
        model = LinearRegression()
        model.fit(self.train_feature, self.train_crimes)
        y_pred = model.predict(self.test_feature)
        r2 = r2_score(self.test_crimes, y_pred)
        print("Linear Regression R^2 Score: " + str(r2))

        if plot:
            pass

    def quadratic_model(self, plot=False):
        model = make_pipeline(PolynomialFeatures(2), LinearRegression())
        model.fit(self.train_feature, self.train_crimes)
        y_pred = model.predict(self.test_feature)
        r2 = r2_score(self.test_crimes, y_pred)
        print("Quadratic Regression R^2 Score: " + str(r2))

        if plot:
            pass

    def cubic_model(self, plot=False):
        model = make_pipeline(PolynomialFeatures(3), LinearRegression())
        model.fit(self.train_feature, self.train_crimes)
        y_pred = model.predict(self.test_feature)
        r2 = r2_score(self.test_crimes, y_pred)
        print("Cubic Regression R^2 Score: " + str(r2))

        if plot:
            pass

    def quartic_model(self, plot=False):
        model = make_pipeline(PolynomialFeatures(4), LinearRegression())
        model.fit(self.train_feature, self.train_crimes)
        y_pred = model.predict(self.test_feature)
        r2 = r2_score(self.test_crimes, y_pred)
        print("Quartic Regression R^2 Score: " + str(r2))

        if plot:
            pass

    def gaussian_model(self, plot=False):
        kernel = DotProduct() + WhiteKernel()
        model = GaussianProcessRegressor(kernel=kernel, random_state=42)
        model.fit(self.train_feature, self.train_crimes)
        y_pred = model.predict(self.test_feature)
        r2 = r2_score(self.test_crimes, y_pred)
        print("Gaussian Regression R^2 Score: " + str(r2))

        if plot:
            pass

    def random_forest_model(self, plot=False):
        model = RandomForestRegressor(n_estimators = 100, max_depth = None, random_state=42)
        model.fit(self.train_feature, self.train_crimes)
        y_pred = model.predict(self.test_feature)
        r2 = r2_score(self.test_crimes, y_pred)
        print("Random Forest Regression R^2 Score: " + str(r2))

        if plot:
            pass

    def xgboost_model(self, plot=False):
        model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
        model.fit(self.train_feature, self.train_crimes)
        y_pred = model.predict(self.test_feature)
        r2 = r2_score(self.test_crimes, y_pred)
        print("XGBoost Regression R^2 Score: " + str(r2))

        if plot:
            pass



class DiscreteAnalyzer(BaseAnalyzer):
    def __init__(self, result_df):
        BaseAnalyzer.__init__(self, result_df)

    def convolve_bins(self, convolution):
        """
        Recompute bin feature values via a convolution filter.
        """

    def box_plotter(self):
        """
        Plot box plots along each value of the discrete variable
        """
