import pandas as pd
import joinaroo
from preprocessors.base import *
from calculations.base import DiscreteCalc, DistCalc
from analysis.base import DiscreteAnalyzer
from time import time

''' Generating csv files for Keis's multiple feature testing script '''

subgroups = ['Theft Crime', 'Violent Crime']

# discrete_info = [
# 		('Liquor', 'data/liquor-licenses.csv', preprocess_liquor),
# 		('Entertainment', 'data/entertainment-licenses.csv', preprocess_entertainment),
# 		('Traffic_Signals', 'data/Traffic_Signals/Traffic_Signals.shp', preprocess_traffic_signal),
# 		('Streetlights', 'data/streetlight-locations.csv', preprocess_streetlight),
# 		('MBTA_Stops', 'data/MBTA_Stops.csv', preprocess_mbta),
# 		('Trees', 'data/Trees/Trees.shp', preprocess_trees),
# ]

# for feature_name, filename, prep in discrete_info:
# 	print(feature_name)
# 	# Preprocess the data
# 	start = time()
# 	df = prep(filename)
# 	binned_crimes = pd.read_csv('crimes_in_bins.csv')

# 	# Calculations
# 	calculator = DiscreteCalc(df, binned_crimes, feature_name=feature_name)
# 	results = calculator.calculation(subgroups, convolve=False, group=True, to_file=True)
# 	end = time()
# 	print('discrete calculations took {} seconds'.format(round(end-start, 2)))


distance_info = [
		('Charging_Stations', 'data/Charging_Stations/Charging_Stations.shp', preprocess_charging_stations),
		('Colleges', 'data/Colleges_and_Universities/Colleges_and_Universities.shp', preprocess_colleges),
		('Pools', 'data/Community_center_pools/Community_center_pools.shp', preprocess_pools),
		('Community_Centers', 'data/Community_Centers/Community_Centers.shp', preprocess_community_centers),
		('Public_Schools', 'data/Public_Schools/Public_Schools.shp', preprocess_public_schools),
		('Non_Public_Schools', 'data/Non_Public_Schools/Non_Public_Schools.shp', preprocess_private_schools),
		('Hospitals', 'data/hospital-locations.csv', preprocess_hospital),
		('Police_Stations', 'data/Boston_Police_Stations/Boston_Police_Stations.shp', preprocess_police_stations)
]

for feature_name, filename, prep in distance_info:
	# Preprocess the data
	start = time()
	df = prep(filename)
	binned_crimes = pd.read_csv('crimes_in_bins.csv')

	# Calculations
	calculator = DistCalc(df, binned_crimes, feature_name=feature_name)
	results = calculator.calculation(subgroups, feature_function=min, group=True, to_file=True)
	end = time()
	print('dist calculations took {} seconds'.format(round(end-start, 2)))







