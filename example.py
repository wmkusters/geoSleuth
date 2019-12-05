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

'''
How to run:
>>> python3 count_analysis.py Stop num_stops C4

	argv[1]: whatever comes before the _Calculation_Results folder
	argv[2]: column name that indicates the count
	argv[3]: type of train/test sets, use C4 for separate train/test average data
	argv[4] (optional): '--plot' to plot the regression results

'''

def count_analysis(data_name, data_col, transform='C4', plot=False):
	violent_crime = pd.read_csv(data_name+'_Calculation_Results/Violent Crime.csv')
	theft_crime = pd.read_csv(data_name+'_Calculation_Results/Theft Crime.csv')
	road_crime = pd.read_csv(data_name+'_Calculation_Results/On Road Crime.csv')
	other_definite_crimes = pd.read_csv(data_name+'_Calculation_Results/Other Definite Crimes.csv')
	other_non_definite_crimes = pd.read_csv(data_name+'_Calculation_Results/Other Non-Definite Crimes.csv')
	all_definite_crimes = pd.read_csv(data_name+'_Calculation_Results/All Definite Crimes.csv')
	all_non_definite_crimes = pd.read_csv(data_name+'_Calculation_Results/All Non-Definite Crimes.csv')

	crime_groupings = [violent_crime, theft_crime, road_crime, other_definite_crimes, other_non_definite_crimes, all_definite_crimes, all_non_definite_crimes]
	crime_groupings_names = ['Violent', 'Theft', 'On Road', 'Other Definite', 'Other Non-Definite', 'All Definite', 'All Non-Definite']

	linear_regression_results = []
	quadratic_regression_results = []
	cubic_regression_results = []
	quadratic_regression_results = []
	quartic_regression_results = []
	gaussian_regression_results = []
	random_forest_regression_results = []
	xgboost_regression_results = []

	i=0
	for crime_group_df in crime_groupings:
		print(crime_groupings_names[i])

		# filter to bins with nonzero crimes
		filtered = crime_group_df[crime_group_df['num_crimes'] > 0]

		# get average
		avg = filtered.groupby(data_col)[['num_crimes']].mean().reset_index()
		
		# C1: use this if training/testing on all data
		num_stops = np.array(filtered[data_col]).reshape(-1, 1)
		num_crimes = np.array(filtered['num_crimes'])

		# C2: use this if training/testing on avg data
		avg_stops = np.array(avg[data_col]).reshape(-1, 1)
		avg_crimes = np.array(avg['num_crimes'])

		# C3: use this if splitting train/test on all data
		train, test = train_test_split(filtered, test_size=0.1, random_state=42)
		train = train.sort_values(data_col)
		test = test.sort_values(data_col)
		train_stops = np.array(train[data_col]).reshape(-1, 1)
		train_crimes = np.array(train['num_crimes'])
		test_stops = np.array(test[data_col]).reshape(-1, 1)
		test_crimes = np.array(test['num_crimes'])

		# C4: use this if splitting train/test on avg data
		train_avg = train.groupby(data_col)[['num_crimes']].mean().reset_index()
		train_avg_stops = np.array(train_avg[data_col]).reshape(-1, 1)
		train_avg_crimes = np.array(train_avg['num_crimes'])
		test_avg = test.groupby(data_col)[['num_crimes']].mean().reset_index()
		test_avg_stops = np.array(test_avg[data_col]).reshape(-1, 1)
		test_avg_crimes = np.array(test_avg['num_crimes'])

		get_sets = {'C1': [num_stops, num_crimes, num_stops, num_crimes],
					'C2': [avg_stops, avg_crimes, avg_stops, avg_crimes],
					'C3': [train_stops, train_crimes, test_stops, test_crimes],
					'C4': [train_avg_stops, train_avg_crimes, test_avg_stops, test_avg_crimes]}

		x_train, y_train, x_test, y_test = get_sets[transform]


		f, ax = plt.subplots(4,2,figsize=(12,8), constrained_layout=True)

		#make initial scatterplot of the data
		ax[0][0].scatter(num_stops, num_crimes)
		ax[0][0].set_title('Number of {} Vs Relative Number of Reported Crimes'.format(data_name.replace('_',' ')))


		#try linear regression and get r^2 score
		model = LinearRegression()

		model.fit(x_train, y_train)
		y_pred = model.predict(x_test)
		linear_r2_score = r2_score(y_test, y_pred)
		linear_regression_results.append(linear_r2_score)
		print("Linear Regression R^2 Score: " + str(linear_r2_score))

		ax[0][1].scatter(x_train, y_train)
		ax[0][1].scatter(x_test, y_pred, color='r')
		ax[0][1].set_title('Linear Regression Prediction (R^2: {})'.format(str(round(linear_r2_score, 2))))


		#try quadratic regression and get r^2 score
		model = make_pipeline(PolynomialFeatures(2), LinearRegression())
		
		model.fit(x_train, y_train)
		y_pred = model.predict(x_test)
		quadratic_r2_score = r2_score(y_test, y_pred)
		quadratic_regression_results.append(quadratic_r2_score)
		print("Quadratic Regression R^2 Score: " + str(quadratic_r2_score))
		
		ax[1][0].scatter(x_train, y_train)
		ax[1][0].scatter(x_test, y_pred, color='r')
		ax[1][0].set_title('Quadratic Regression Prediction (R^2: {})'.format(str(round(quadratic_r2_score, 2))))


		#try cubic regression and get r^2 score
		model = make_pipeline(PolynomialFeatures(3), LinearRegression())

		model.fit(x_train, y_train)
		y_pred = model.predict(x_test)
		cubic_r2_score = r2_score(y_test, y_pred)
		cubic_regression_results.append(cubic_r2_score)
		print("Cubic Regression R^2 Score: " + str(cubic_r2_score))
		
		ax[1][1].scatter(x_train, y_train)
		ax[1][1].scatter(x_test, y_pred, color='r')
		ax[1][1].set_title('Cubic Regression Prediction (R^2: {})'.format(str(round(cubic_r2_score, 2))))


		#try quartic regression and get r^2 score
		model = make_pipeline(PolynomialFeatures(4), LinearRegression())
		model.fit(x_train, y_train)
		y_pred = model.predict(x_test)
		quartic_r2_score = r2_score(y_test, y_pred)
		quartic_regression_results.append(quartic_r2_score)
		print("Quartic Regression R^2 Score: " + str(quartic_r2_score))
		
		ax[2][0].scatter(x_train, y_train)
		ax[2][0].scatter(x_test, y_pred, color='r')
		ax[2][0].set_title('Quartic Regression Prediction (R^2: {})'.format(str(round(quartic_r2_score, 2))))


		#try gaussian regressor and get r^2 score
		kernel = DotProduct() + WhiteKernel()
		model = GaussianProcessRegressor(kernel=kernel, random_state=42)
		model.fit(x_train, y_train)
		y_pred = model.predict(x_test)
		gaussian_r2_score = r2_score(y_test, y_pred)
		gaussian_regression_results.append(gaussian_r2_score)
		print("Gaussian Regression R^2 Score: " + str(gaussian_r2_score))

		ax[2][1].scatter(x_train, y_train)
		ax[2][1].scatter(x_test, y_pred, color='r')
		ax[2][1].set_title('Gaussian Regression Prediction (R^2: {})'.format(str(round(gaussian_r2_score, 2))))


		#try random forest regressor and get r^2 score
		model = RandomForestRegressor(n_estimators = 100, max_depth = None, random_state=42)
		model.fit(x_train, y_train)
		y_pred = model.predict(x_test)
		random_forest_r2_score = r2_score(y_test, y_pred)
		random_forest_regression_results.append(random_forest_r2_score)
		print("Random Forest Regression R^2 Score: " + str(random_forest_r2_score))
		
		ax[3][0].scatter(x_train, y_train)
		ax[3][0].scatter(x_test, y_pred, color='r')
		ax[3][0].set_title('Random Forest Regression Prediction (R^2: {})'.format(str(round(random_forest_r2_score, 2))))

		#try xgboost regressor and get r^2 score
		model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
		model.fit(x_train, y_train)
		y_pred = model.predict(x_test)
		xgboost_r2_score = r2_score(y_test, y_pred)
		xgboost_regression_results.append(xgboost_r2_score)
		print("XGBoost Regression R^2 Score: " + str(xgboost_r2_score))
		
		ax[3][1].scatter(x_train, y_train)
		ax[3][1].scatter(x_test, y_pred, color='r')
		ax[3][1].set_title('XGBoost Regression Prediction (R^2: {})'.format(str(round(xgboost_r2_score, 2))))


		f.suptitle('{} Data on {} Crimes using {}'.format(data_name, crime_groupings_names[i], transform))
		if plot == True:
			plt.show()
		i += 1


	results_dict = {'Crime Group': crime_groupings_names,
					'Linear': linear_regression_results, 
					'Quadratic': quadratic_regression_results, 
					'Cubic': cubic_regression_results,
					'Quartic': quartic_regression_results, 
					'Gaussian': gaussian_regression_results, 
					'Random Forest': random_forest_regression_results, 
					'XGBoost': xgboost_regression_results} 

	results_df = pd.DataFrame.from_dict(results_dict)
	results_df.set_index('Crime Group')
	results_df.to_csv('Count_Analysis_Results/{}_Crime_Results.csv'.format(data_name))




if __name__ == '__main__':
	data_name = sys.argv[1]
	data_col = sys.argv[2]
	transform = sys.argv[3]
	plot = ('--plot' in sys.argv)
	count_analysis(data_name, data_col, transform, plot)

