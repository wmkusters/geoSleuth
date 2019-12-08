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
import os


class BaseCalc:
    def __init__(self, feature_df, binned_crime_df, feature_name=None):
        self.feature_df = feature_df
        self.binned_crime_df = binned_crime_df
        self.feature_name = feature_name

    def calculation():
        raise NotImplementedError("Calculations are implemented in specific class!")

    def grouper(self, result_df):
        """
        parameters:
            result_df: binned crime df with feature values calculated and appended
        returns:
            grouped

        """
        features = result_df.groupby("bin_id").first()["feature"]
        crimes = result_df.groupby("bin_id").count()["feature"]
        area_proportion = result_df.groupby("bin_id").first()["area_proportion"]
        result_df = (
            pd.merge(features, crimes, on="bin_id")
            .reset_index()
            .rename(columns={"feature_x": "feature", "feature_y": "num_crimes"})
        )
        result_df = pd.merge(result_df, area_proportion, on="bin_id")
        return result_df

    def write_results(self, result_dict):
        assert self.feature_name is not None
        result_dir = self.feature_name + "_Results/"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for subgroup in result_dict.keys():
            result_dict[subgroup].to_csv(
                result_dir + subgroup.replace(" ", "") + ".csv"
            )


class DistCalc(BaseCalc):
    """
    Base class for building dataframes mapping
    bins to distances away from certain features
    """

    def __init__(self, feature_df, binned_crime_df):
        """
        parameters:
            feature_df: preprocessed PANDAS (not geopandas) dataframe of a feature
            binned_crime_df: preprocessed binned crime pandas dataframe 
        returns:
            class instance

        Pandas dataframes are used as data is read from a .csv, not from a .shp, so no
        geometry column is set.
        """
        BaseCalc.__init__(self, feature_df, binned_crime_df)

    def calculation(
        self, subgroup_list, feature_function=min, group=False, to_file=False
    ):
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
         The resulting dataframes would be composed of the same rows as self.binned_crime_df,
         with a new final column that is the value of a feature for the bin of that row.
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
            cent_coords = wkt.loads(  # Using wkt.loads() as points are stored as strings
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


class DiscreteCalc(BaseCalc):
    """
    Base class for building dataframes mapping
    bins to count of discrete instances of 
    certain features within a bin
    """

    def __init__(self, feature_df, binned_crime_df, feature_name):
        """
        parameters:
            feature_df: preprocessed PANDAS (not geopandas) dataframe of a feature
            binned_crime_df: preproccessed binned crime pandas dataframe 
        returns:
            class instance

        Pandas dataframes are used, as data is read from a .csv, not from a .shp, so no
        geometry column is set.
        """
        BaseCalc.__init__(self, feature_df, binned_crime_df, feature_name)

    def calculation(self, subgroup_list, convolve=False, group=False, to_file=False):
        """
        parameters:
            subgroup_list: list of subgroups to return data for
        returns:
            list of dataframes containing bins/crimes, one for each passed
            subgroup, with feature values appended as the last column

        This method returns a list of dataframes much the same as the DistCalc
        class. Based on a series of table operations, a column is added to the
        end of binned crime dataframe labeled "feature" which stores the value
        for the calculated feature for that row's bin, the same as DistCalc.
        """
        # Compose gdf with just unique bins and their geometries for joining
        bins = (
            self.binned_crime_df.groupby("bin_id")["geometry"]
            .apply(lambda poly: wkt.loads(np.unique(poly)[0]))
            .reset_index()
        )
        crs = {"init": "epsg:4326"}
        bins = gpd.GeoDataFrame(bins, crs=crs, geometry=bins.geometry)

        # Join the features onto the bins, based on the feature (i.e. a liquor
        # license) being inside of the bin
        feature_calculation = gpd.sjoin(bins, self.feature_df, op="contains")
        feature_calculation = (
            feature_calculation.groupby("bin_id").count().reset_index()
        )

        # Rename feature column to feature
        common_cols = {"bin_id", "geometry", "index_right"}
        for col in feature_calculation.columns:
            if col not in common_cols:
                feature_col = col
        feature_calculation = feature_calculation.rename(
            columns={feature_col: "feature"}
        )

        # Join the features onto the original binned crime dataframe
        feat_result_df = pd.merge(
            self.binned_crime_df,
            feature_calculation[["bin_id", "feature"]],
            on="bin_id",
        )

        # Filter by subgroup, return filtered results
        results = {}
        for subgroup in subgroup_list:
            crimes = subgroups[subgroup]  # Add try catch for not using correct data
            subgroup_df = feat_result_df[
                feat_result_df["OFFENSE_CODE_GROUP"].isin(crimes)
            ]

            # If group=True, group the subgroup dataframe into bin_ids and
            # the relevant columns
            if group:
                grouped_df = self.grouper(subgroup_df)
                results[subgroup] = grouped_df
            else:
                results[subgroup] = subgroup_df

        if to_file:
            self.write_results(results)

        return results
