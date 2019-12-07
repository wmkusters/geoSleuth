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
        parameters:
            filename: location of csv file to analyze
        returns:
            class instance
        """

        self.result_df = pd.read_csv(filename)
        # @bill: use this to generate dataframe of form | bin_id | feature | num_crimes | area_proportion |
        # and save to csv
        ###################
        features = self.result_df.groupby("bin_id").first()["feature"]
        crimes = self.result_df.groupby("bin_id").count()["feature"]
        area_proportion = self.result_df.groupby("bin_id").first()["area_proportion"]
        self.result_df = pd.merge(features, crimes, on="bin_id")
                         .reset_index()
                         .rename(columns={"feature_x": "feature", "feature_y": "num_crimes"})
        self.result_df = pd.merge(self.result_df, area_proportion, on="bin_id").reset_index()
        ###################

        # filter to bins with nonzero crimes
        self.result_df = self.result_df[self.result_df["num_crimes"] > 0]

        # split train/test
        self.train, self.test = train_test_split(
            self.result_df, test_size=0.1, random_state=42
        )
        # train = train.sort_values('feature')
        # test = test.sort_values('feature')
        self.train_x = np.array(train["feature"]).reshape(-1, 1)
        self.train_y = np.array(train["num_crimes"])
        self.test_x = np.array(test["feature"]).reshape(-1, 1)
        self.test_y = np.array(test["num_crimes"])

    def linear_model(self, plot=False):
        model = LinearRegression()
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        r2 = r2_score(self.test_y, y_pred)
        print("Linear Regression R^2 Score: " + str(r2))

        if plot:
            self.plot_result('Linear', y_pred, r2)

    def quadratic_model(self, plot=False):
        model = make_pipeline(PolynomialFeatures(2), LinearRegression())
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        r2 = r2_score(self.test_y, y_pred)
        print("Quadratic Regression R^2 Score: " + str(r2))

        if plot:
            self.plot_result('Quadratic', y_pred, r2)

    def cubic_model(self, plot=False):
        model = make_pipeline(PolynomialFeatures(3), LinearRegression())
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        r2 = r2_score(self.test_y, y_pred)
        print("Cubic Regression R^2 Score: " + str(r2))

        if plot:
            self.plot_result('Cubic', y_pred, r2)

    def quartic_model(self, plot=False):
        model = make_pipeline(PolynomialFeatures(4), LinearRegression())
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        r2 = r2_score(self.test_y, y_pred)
        print("Quartic Regression R^2 Score: " + str(r2))

        if plot:
            self.plot_result('Quartic', y_pred, r2)

    def gaussian_model(self, plot=False):
        kernel = DotProduct() + WhiteKernel()
        model = GaussianProcessRegressor(kernel=kernel, random_state=42)
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        r2 = r2_score(self.test_y, y_pred)
        print("Gaussian Regression R^2 Score: " + str(r2))

        if plot:
            self.plot_result('Gaussian', y_pred, r2)

    def random_forest_model(self, plot=False):
        model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        r2 = r2_score(self.test_y, y_pred)
        print("Random Forest Regression R^2 Score: " + str(r2))

        if plot:
            self.plot_result('Random Forest', y_pred, r2)

    def xgboost_model(self, plot=False):
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            colsample_bytree=0.3,
            learning_rate=0.1,
            max_depth=5,
            alpha=10,
            n_estimators=10,
        )
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        r2 = r2_score(self.test_y, y_pred)
        print("XGBoost Regression R^2 Score: " + str(r2))

        if plot:
            self.plot_result('XGBoost', y_pred, r2)

    def plot_result(self, model, y_pred, r2):
        plt.scatter(self.train_x, self.train_y)
        plt.scatter(self.test_x, y_pred, color='r')
        plt.title('{} Regression Prediction (R^2: {})'.format(model, str(round(r2, 2))))
        plt.show()


class DiscreteAnalyzer(BaseAnalyzer):
    """
    Analyzer for discrete features
    """
    def __init__(self, result_df):
        BaseAnalyzer.__init__(self, result_df)

        # average bins
        # train_avg = self.train.groupby("feature")[["num_crimes"]].mean().reset_index()
        # test_avg = self.test.groupby("feature")[["num_crimes"]].mean().reset_index()

        # # this overrides the generic train/test sets generated in BaseAnalyzer
        # self.train_x = np.array(train_avg["feature"]).reshape(-1, 1)
        # self.train_y = np.array(train_avg["num_crimes"])
        # self.test_x = np.array(test_avg["feature"]).reshape(-1, 1)
        # self.test_y = np.array(test_avg["num_crimes"])

    def convolve_bins(self, convolution):
        """
        parameters: 
            convolution: the operation to be applied during convolution, i.e.
                         averaging
        returns: 
            the same dataframe with the feature values recomputed
            by the convolution

        Recompute bin feature values via a convolution filter. Uses a constant
        dict provided in the library mapping a bin_id to the ids of adjacent
        bins for faster computation of the convolution.
        """

        # BASICALLY PSEUDOCODE RN
        def pandas_convolution(dataframe):
            adj_values = [dataframe.at[bin_id, feature] for bin_id in adj_dict[bin_id]]
            if dataframe.at[bin_id, feature] > 0:
                adj_values.append(dataframe.at[bin_id, feature])
            dataframe.at[bin_id, feature] = func(adj_values)

        raise NotImplementedError()

    def box_plotter(self):
        """
        Plot box plots along each value of the discrete variable
        """
        self.result_df.boxplot('num_crimes', by='feature')
        plt.show()
