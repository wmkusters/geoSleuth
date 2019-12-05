from math import radians, cos, sin, asin, sqrt
from shapely.geometry import Point, Polygon
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from geopandas import GeoSeries
from shapely.geometry.multipolygon import MultiPolygon
from shapely import wkt as wkt
from vars.subgroupings import subgroups


class BaseCalc:
    def __init__(self, feature_df, binned_crime_df):
        self.feature_df = feature_df
        self.binned_crime_df = binned_crime_df

    def calculation():
        raise NotImplementedError("Calculations are implemented in specific class!")


class DistCalc(BaseCalc):
    """
    Base class for building dataframes mapping
    bins to distances away from certain features
    """

    def __init__(self, feature_df, binned_crime_df):
        """
        parameters:
            feature_df: preprocessed PANDAS (not geopandas) dataframe of a feature
            binned_crime_df: preproccessed binned crime pandas dataframe 
        returns:
            class instance

        Pandas dataframes are used as data is read from a .csv, not from a .shp, so no
        geometry column is set.
        """
        BaseCalc.__init__(self, feature_df, binned_crime_df)

    def calculation(self, subgroup_list, feature_function):
        """
        parameters:
            subgroup list: list of subgroups to independently calculate/return
            feature function: function used on feature for calculation
        returns:
            list of resulting dataframes

         This method performs a distance feature calculation, calculating distances 
         of each bin to some feature with a location, i.e. bins to hospitals. With 
         that list of distances, it then maps each bin to a function performed over 
         those lists (feature_function parameter). An example function would be
         summing the inverse squares of the distances from bin centroid to each feature.
        """

        def haversine(coord_tuple):
            """
            parameters:
                coord_tuple: string, contains (coord 1) and (coord 2) in tuple format
                             concatenated together
            returns:
                haversine distance between coord 1 and coord 2
            """
            (lon1, lat1, lon2, lat2) = coord_tuple
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * asin(sqrt(a))
            r = 3956  # Radius of earth in miles. Use 6371 for kilometers
            return c * r

        # Get list of unique bin ids for distance mapping
        bin_ids = pd.unique(self.binned_crime_df.bin_id)

        # Iterate through unique bins, calculating list of distances
        # from that bin's centroid to each feature's coordinates,
        # map bin_id to variable function of that distance list
        bin_distances = {}
        for bin_id in bin_ids:
            cent_coords = wkt.loads(
                self.binned_crime_df.loc[self.binned_crime_df["bin_id"] == bin_id]
                .iloc[0]
                .centroid
            )
            cent_coords = (cent_coords.x, cent_coords.y)
            distances = []
            for point in self.feature_df.geometry:
                point = wkt.loads(point)
                feat_coords = (point.x, point.y)
                distances.append(haversine(cent_coords + feat_coords))
            bin_distances[bin_id] = feature_function(distances)

        # Add the result to the dataframe as a column
        self.binned_crime_df["feature"] = self.binned_crime_df.apply(
            lambda row: bin_distances[row["bin_id"]], axis=1
        )

        # Return results filtered for each subgroup of crime
        results = []
        for subgroup in subgroup_list:
            crimes = subgroups[subgroup]
            filtered_df = self.binned_crime_df[
                self.binned_crime_df["OFFENSE_CODE_GROUP"].isin(crimes)
            ]
            results.append(filtered_df)
        return results


def CountCalc(BaseCalc):
    def __init__():
        pass

    def calculation(self, filename):
        liquor_data = gpd.read_file(filename)
        liquor_data["Long"] = liquor_data["Long"].astype(float)
        liquor_data["Lat"] = liquor_data["Lat"].astype(float)
        liquor_geometry = [
            Point(xy) for xy in zip(liquor_data["Long"], liquor_data["Lat"])
        ]
        liquor_data = gpd.GeoDataFrame(liquor_data, crs=crs, geometry=liquor_geometry)

        crime_bins = gpd.read_file("")
        result = pd.Dataframe()
        for subgroup in subgroups:
            for bin_id in crime_bins.OBJECTID.unique:
                num_licenses = liquor_data[liquor_data.within(crime_bins)].count()
