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

    def __init__(self, result_df):
        """
        Prepare df for analysis (train test split)
        """
        self.result_df = result_df

    def linear_model(self, plot=False):
        pass

    def quadratic_model(self, plot=False):
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
