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
    def __init__(self, feature_df, binned_crime_df):
        BaseCalc.__init__(self, feature_df, binned_crime_df)

    def calculation(self, subgroup_list):
        def haversine(coord_tuple):
            (lon1, lat1, lon2, lat2) = coord_tuple
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * asin(sqrt(a))
            r = 3956  # Radius of earth in miles. Use 6371 for kilometers
            return c * r

        results = []
        for subgroup in subgroup_list:
            crimes = subgroups[subgroup]
            filtered_df = self.binned_crime_df[
                self.binned_crime_df["OFFENSE_CODE_GROUP"].isin(crimes)
            ]
            bin_ids = pd.unique(self.binned_crime_df.bin_id)
            bin_distances = {}
            for bin_id in bin_ids:
                cent_coords = (
                    filtered_df.loc[filtered_df["bin_id"] == bin_id].iloc[0].centroid
                )

                distances = []
                for point in self.feature_df.geometry:
                    print(point)
                    point = wkt.loads(point)
                    print(point)
                    print("------")
                    hospital_coords = (
                        point.x + point.geometry.y
                    )
                    distances.append(haversine(cent_coords + hospital_coords))
                print(distances)
                raise SystemError(0)
                feature_value = min(
                    [dist for dist in distances]
                )
                filtered_df.at(bin_id)[feature_name] = feature_value
            filtered_df.groupby("bin_id").count()
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
