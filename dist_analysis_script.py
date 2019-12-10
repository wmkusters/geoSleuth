import joinaroo
from preprocessors.base import *
from calculations.base import DistCalc
from analysis.base import BaseAnalyzer
import pandas as pd
from time import time

''' The whole pipeline '''
# Warning!! This takes a while to run (5~10 mins).
# You can comment out joinaroo if crime_in_bins.csv already exists

start = time()
joinaroo.main()
end = time()
print('joinaroo took {} seconds'.format(round(end-start, 2)))

# Set up datasets: (feature name, filename, preprocessor class)
datasets = [('Charging_Stations', 'data/Charging_Stations/Charging_Stations.shp', BaseProcessor),
			('Colleges', 'data/Colleges_and_Universities/Colleges_and_Universities.shp', BaseProcessor),
			('Pools', 'data/Community_center_pools/Community_center_pools.shp', BaseProcessor),
			('Community_Centers', 'data/Community_Centers/Community_Centers.shp', BaseProcessor),
			('Public_Schools', 'data/Public_Schools/Public_Schools.shp', BaseProcessor),
			('Non_Public_Schools', 'data/Non_Public_Schools/Non_Public_Schools.shp', BaseProcessor),
			('Hospitals', 'data/hospital-locations.csv', HospitalProcessor),
			('Police_Stations', 'data/Boston_Police_Stations/Boston_Police_Stations.shp', BaseProcessor)]

# Crime subgroupings to analyze
subgroups = ["Violent Crime","Theft Crime","On Road Crime","Other Definite Crimes","Other Non-Definite Crimes","All Definite Crimes","All Non-Definite Crimes"]

for feature_name, filename, prep in datasets:
	# Preprocess the data
	start = time()
	df = prep(filename).process()
	binned_crimes = pd.read_csv('crimes_in_bins.csv')

	# Calculations
	calculator = DistCalc(df, binned_crimes, feature_name=feature_name)
	results = calculator.calculation(subgroups, feature_function=min, group=True, to_file=True)
	end = time()
	print('calculations took {} seconds'.format(round(end-start, 2)))

	# Analysis
	start = time()
	for subgroup in subgroups:
		analyzer = BaseAnalyzer(results[subgroup])
		analyzer.run_models(plot=False)
	end = time()
	print('analysis took {} seconds'.format(round(end-start, 2)))
