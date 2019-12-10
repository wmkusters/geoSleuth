import joinaroo
from preprocessors.base import *
from calculations.base import DiscreteCalc
from analysis.base import DiscreteAnalyzer
import pandas as pd
from time import time

''' The whole pipeline '''
# Warning!! This takes a while to run (5~10 mins)
# You can comment out joinaroo if crime_in_bins.csv already exists

start = time()
joinaroo.main()
end = time()
print('joinaroo took {} seconds'.format(round(end-start, 2)))

# Set up datasets: (feature name, filename, preprocessor class)
datasets = [('Liquor', 'data/liquor-licenses.csv', LiquorProcessor),
			('Entertainment', 'data/entertainment-licenses.csv', EntertainmentProcessor),
			('Traffic_Signals', 'data/Traffic_Signals/Traffic_Signals.shp', BaseProcessor),
			('Streetlights', 'data/streetlight-locations.csv', StreetlightProcessor),
			('MBTA_Stops', 'data/MBTA_Stops.csv', MbtaProcessor),
			('Trees', 'data/Trees/Trees.shp', BaseProcessor)]

# Crime subgroupings to analyze
subgroups = ["Violent Crime","Theft Crime","On Road Crime","Other Definite Crimes","Other Non-Definite Crimes","All Definite Crimes","All Non-Definite Crimes"]

for feature_name, filename, prep in datasets:
	# Preprocess the data
	start = time()
	df = prep(filename).process()
	binned_crimes = pd.read_csv('crimes_in_bins.csv')

	# Calculations
	calculator = DiscreteCalc(df, binned_crimes, feature_name=feature_name)
	# TODO setting convolve=True gives me a "ValueError: Input contains NaN, infinity or a value too large for dtype('float64')" error in analysis
	results = calculator.calculation(subgroups, convolve=False, group=True, to_file=False)
	end = time()
	print('calculations took {} seconds'.format(round(end-start, 2)))

	# Analysis
	start = time()
	for subgroup in subgroups:
		analyzer = DiscreteAnalyzer(results[subgroup])
		# analyzer.box_plotter()
		analyzer.run_models(plot=False)
	end = time()
	print('analysis took {} seconds'.format(round(end-start, 2)))
