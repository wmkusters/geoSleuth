import pandas as pd
import joinaroo
from preprocessors.base import *
from calculations.base import DiscreteCalc
from analysis.base import DiscreteAnalyzer
from time import time

''' The whole pipeline '''

start = time()
# joinaroo.main()
end = time()
print('joinaroo took {} seconds'.format(round(end-start, 2)))

# info = [('Liquor', 'data/liquor-licenses.csv', preprocess_liquor)]
info = [('MBTA_Stops', 'data/MBTA_Bus_Stops.csv', preprocess_mbta)]
# subgroups = ["Violent Crime","Theft Crime","On Road Crime","Other Definite Crimes","Other Non-Definite Crimes","All Definite Crimes","All Non-Definite Crimes"]
subgroups = ['Theft Crime', 'All Definite Crimes']

for feature_name, filename, prep in info:
	# Preprocess the data
	# TODO this might change once preprocessor class is made
	start = time()
	df = prep(filename)
	binned_crimes = pd.read_csv('crimes_in_bins.csv')

	# Calculations
	calculator = DiscreteCalc(df, binned_crimes, feature_name=feature_name)
	# TODO setting convolve=True gives me a "ValueError: Input contains NaN, infinity or a value too large for dtype('float64')" error in analysis
	results = calculator.calculation(subgroups, convolve=False, group=True, to_file=True)
	end = time()
	print('calculations took {} seconds'.format(round(end-start, 2)))

	# Analysis
	start = time()
	for subgroup in subgroups:
		print(subgroup)
		analyzer = DiscreteAnalyzer(results[subgroup])
		analyzer.box_plotter()
		# run models and save to csv; TODO will be abstracted
		analyzer.linear_model()
		analyzer.quadratic_model()
		analyzer.cubic_model()
		analyzer.quartic_model()
		analyzer.gaussian_model()
		analyzer.random_forest_model()
		analyzer.xgboost_model()
		print('\n')
	end = time()
	print('analysis took {} seconds'.format(round(end-start, 2)))