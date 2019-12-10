import pandas as pd
from preprocessors.base import *

# Test BaseProcessor
filename = 'data/Charging_Stations/Charging_Stations.shp'
df = BaseProcessor(filename).process()
# check that it returns a GeoDataFrame
assert (type(df) == gpd.GeoDataFrame)
# check that it doesn't contain empty geometry
assert (sum(df.geometry.isna() | df.geometry.is_empty) == 0)


# Test Processors for csv
filename = 'data/liquor-licenses.csv'
df = LiquorProcessor(filename).process()
# check that it returns a GeoDataFrame
assert (type(df) == gpd.GeoDataFrame)
# check that it doesn't contain empty geometry
assert (sum(df.geometry.isna() | df.geometry.is_empty) == 0)

filename = 'data/entertainment-licenses.csv'
df = EntertainmentProcessor(filename).process()
# check that it returns a GeoDataFrame
assert (type(df) == gpd.GeoDataFrame)
# check that it doesn't contain empty geometry
assert (sum(df.geometry.isna() | df.geometry.is_empty) == 0)

filename = 'data/streetlight-locations.csv'
df = StreetlightProcessor(filename).process()
# check that it returns a GeoDataFrame
assert (type(df) == gpd.GeoDataFrame)
# check that it doesn't contain empty geometry
assert (sum(df.geometry.isna() | df.geometry.is_empty) == 0)

filename = 'data/MBTA_Stops.csv'
df = MbtaProcessor(filename).process()
# check that it returns a GeoDataFrame
assert (type(df) == gpd.GeoDataFrame)
# check that it doesn't contain empty geometry
assert (sum(df.geometry.isna() | df.geometry.is_empty) == 0)

filename = 'data/hospital-locations.csv'
df = HospitalProcessor(filename).process()
# check that it returns a GeoDataFrame
assert (type(df) == gpd.GeoDataFrame)
# check that it doesn't contain empty geometry
assert (sum(df.geometry.isna() | df.geometry.is_empty) == 0)
