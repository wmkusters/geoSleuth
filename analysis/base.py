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

    def __init__(self, dataframe, feature_name=None):
        """
        parameters:
            dataframe: pandas dataframe to analyze. must contain 'bin_id', 'feature', 'num_crimes'
        returns:
            class instance
        """

        self.result_df = dataframe

        # sanity check
        assert "bin_id" in self.result_df.columns
        assert "feature" in self.result_df.columns
        assert "num_crimes" in self.result_df.columns

        # filter to bins with nonzero crimes
        self.result_df = self.result_df[self.result_df["num_crimes"] > 0]

        # split train/test
        self.train, self.test = train_test_split(self.result_df, random_state=42)
        self.train_x = np.array(self.train["feature"]).reshape(-1, 1)
        self.train_y = np.array(self.train["num_crimes"])
        self.test_x = np.array(self.test["feature"]).reshape(-1, 1)
        self.test_y = np.array(self.test["num_crimes"])
        self.feature_name = feature_name

    def linear_model(self, plot=False):
        model = LinearRegression()
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        r2 = r2_score(self.test_y, y_pred)
        print("Linear Regression R^2 Score: " + str(r2))

        if plot:
            self.plot_result("Linear", y_pred, r2)
        return r2

    def quadratic_model(self, plot=False):
        model = make_pipeline(PolynomialFeatures(2), LinearRegression())
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        r2 = r2_score(self.test_y, y_pred)
        print("Quadratic Regression R^2 Score: " + str(r2))

        if plot:
            self.plot_result("Quadratic", y_pred, r2)
        return r2

    def cubic_model(self, plot=False):
        model = make_pipeline(PolynomialFeatures(3), LinearRegression())
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        r2 = r2_score(self.test_y, y_pred)
        print("Cubic Regression R^2 Score: " + str(r2))

        if plot:
            self.plot_result("Cubic", y_pred, r2)
        return r2

    def quartic_model(self, plot=False):
        model = make_pipeline(PolynomialFeatures(4), LinearRegression())
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        r2 = r2_score(self.test_y, y_pred)
        print("Quartic Regression R^2 Score: " + str(r2))

        if plot:
            self.plot_result("Quartic", y_pred, r2)
        return r2

    def gaussian_model(self, plot=False):
        kernel = DotProduct() + WhiteKernel()
        model = GaussianProcessRegressor(kernel=kernel, random_state=42)
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        r2 = r2_score(self.test_y, y_pred)
        print("Gaussian Regression R^2 Score: " + str(r2))

        if plot:
            self.plot_result("Gaussian", y_pred, r2)
        return r2

    def random_forest_model(self, plot=False):
        model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        r2 = r2_score(self.test_y, y_pred)
        print("Random Forest Regression R^2 Score: " + str(r2))

        if plot:
            self.plot_result("Random Forest", y_pred, r2)
        return r2

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
            self.plot_result("XGBoost", y_pred, r2)
        return r2

    def plot_result(self, model, y_pred, r2):
        # TODO show predictions for all x, not just test
        plt.scatter(self.train_x, self.train_y)
        plt.scatter(self.test_x, y_pred, color="r")
        plt.title("{} Regression Prediction (R^2: {})".format(model, str(round(r2, 2))))
        plt.show()

    def plot_raw(self, plot_type, **kwargs):
        """
        params:
            plot_type: method name to call for specific
                       plot type
            kwargs: keyword arguments for a specific plot
                    type
        returns:
            Axes, subplot of that plot type 
        """
        # Dict mapping string passed in to method
        plot_dict = {
            "histogram": self.feat_histogram_plot,
            "scatter": self.scatter_plot,
            "agg": self.agg_plot,
        }

        assert plot_type in plot_dict.keys(), (
            "Given a not implemented plot type! Implemented plot types include: "
            + str(list(plot_dict.keys())),
        )
        return plot_dict[plot_type](kwargs)

    def feat_histogram_plot(self, param_dict):
        # bin_ranges = param_dict["bin_ranges"]

        if "bins" in param_dict:
            self.result_df.hist(bins=param_dict["bins"])
            plt.show()
        else:
            self.result_df.hist()
            plt.show()

    def agg_plot(self, param_dict):
        """
        params:
            param_dict: kwargs passed in from plot_raw. 
                        Must include a y_var=a column of
                        self.result_df
        returns:
            Axes and pyplot objects with the agg plot.
        Generates an "aggregation" plot, which bins based on the
        feature value of a dataframe and then sums the y values
        in that bin.
        """
        # Need a y variable for plotting
        assert "y_var" in param_dict, (
            "No y variable set for agg plot! Set variable "
            "y_var = a column of the result dataframe."
        )
        # Split features into bins, sum y values within bins
        bins = param_dict["bins"]
        agg_df = self.result_df.groupby(
            pd.cut(self.result_df["feature"], bins=bins)
        ).sum()
        ax = agg_df.plot.bar(y=param_dict["y_var"], rot=0, width=0.98, cmap="coolwarm")

        # Format x-tick strings
        ax.set_xticklabels(
            [
                str(c)[1:-1].replace(",", "-").replace(" ", "\n")
                for c in agg_df.index.categories
            ]
        )
        return plt, ax

    def scatter_plot(self, param_dict):
        """
        """
        plt.scatter(
            self.result_df.feature, self.result_df.num_crimes, alpha=0.75, c="b"
        )
        plt.xlabel(self.feature_name)
        plt.ylabel("Num Crimes in Bin")
        plt.show()

    # method for running all models
    def run_models(
        self,
        models=[
            "linear",
            "quadratic",
            "cubic",
            "quartic",
            "gaussian",
            "random_forest",
            "xgboost",
        ],
        plot=False,
    ):
        result_dict = {}
        for m in models:
            r2 = eval("self.{}_model(plot={})".format(m, plot))
            result_dict[m] = r2
        return result_dict

    # def write_results(self, result_dict):
    #     assert self.feature_name is not None
    #     result_dir = self.feature_name + '_Analysis_Results.csv'

    #     for subgroup in result_dict.keys():
    #         result_dict[subgroup].to_csv(
    #             result_dir + subgroup.replace(' ', '') + '.csv'
    #         )

    #     results_dict = {'Crime Group': crime_groupings_names,
    #                   'Linear': linear_regression_results,
    #                   'Quadratic': quadratic_regression_results,
    #                   'Cubic': cubic_regression_results,
    #                   'Quartic': quartic_regression_results,
    #                   'Gaussian': gaussian_regression_results,
    #                   'Random Forest': random_forest_regression_results,
    #                   'XGBoost': xgboost_regression_results}

    #     results_df = pd.DataFrame.from_dict(results_dict)
    #     results_df.set_index('Crime Group')
    #     results_df.to_csv('Analysis_Results/{}_Crime_Results.csv'.format(data_name))


class DiscreteAnalyzer(BaseAnalyzer):
    """
    Analyzer for discrete features
    """

    def __init__(self, result_df, average=False):
        BaseAnalyzer.__init__(self, result_df)

        if average:
            # average bins to use for analysis
            train_avg = (
                self.train.groupby("feature")[["num_crimes"]].mean().reset_index()
            )
            test_avg = self.test.groupby("feature")[["num_crimes"]].mean().reset_index()

            # this overrides the generic train/test sets generated in BaseAnalyzer
            self.train_x = np.array(train_avg["feature"]).reshape(-1, 1)
            self.train_y = np.array(train_avg["num_crimes"])
            self.test_x = np.array(test_avg["feature"]).reshape(-1, 1)
            self.test_y = np.array(test_avg["num_crimes"])

    def box_plotter(self):
        """
        Plot box plots along each value of the discrete variable
        """
        self.result_df.boxplot("num_crimes", by="feature")
        plt.show()
