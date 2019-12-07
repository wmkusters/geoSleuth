import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt
import numpy as np

def wkt_pt_conversion(pt_string):
    """
    parameters:
        pt_string: string describing a point to be converted
                   to well-known text format currently just
                   converts points enclosed in () with a comma
                   and need to have the Lat/Long parsed into
                   x/y.
    returns
        Null value in the case of a point == (0, 0), or the
        shapely POINT object of the wkt point.
    """
    pt_string = pt_string.replace("(", "")
    pt_string = pt_string.replace(")", "")
    pt_string = pt_string.replace(" ", "")
    pt_string = pt_string.split(",")
    pt_string = "(" + pt_string[1] + " " + pt_string[0] + ")"
    to_wkt = wkt.loads("POINT " + pt_string)
    if to_wkt.x == 0.0 or to_wkt.y == 0.0:
        return np.nan
    return to_wkt

def preprocess_hospital(filename):
    hospital_data = pd.read_csv(filename)
    crs = {"init": "epsg:4326"}
    hospital_data.XCOORD = hospital_data.XCOORD.apply(lambda x: x / -10e5)
    hospital_data.YCOORD = hospital_data.YCOORD.apply(lambda y: y / 10e5)
    hospital_geometry = [
        Point(xy) for xy in zip(hospital_data.XCOORD, hospital_data.YCOORD)
    ]
    hospital_data = gpd.GeoDataFrame(hospital_data, crs=crs, geometry=hospital_geometry)
    hospital_data = hospital_data.dropna()
    return hospital_data


def preprocess_liquor(filename):
    liquor_data = pd.read_csv(filename)
    crs = {"init": "epsg:4326"}
    liquor_data["geometry"] = liquor_data.Location.apply(wkt_pt_conversion)
    liquor_data = liquor_data.dropna(subset=["geometry"])
    liquor_data = gpd.GeoDataFrame(liquor_data, crs=crs, geometry=liquor_data.geometry)
    return liquor_data[["LICENSENO", "geometry"]]


def preprocess_entertainment(filename):
    entertainment_data = gpd.read_file(filename)
    crs = {"init": "epsg:4326"}
    entertainment_data["Long"] = entertainment_data["Long"].astype(float)
    entertainment_data["Lat"] = entertainment_data["Lat"].astype(float)
    entertainment_geometry = [
        Point(xy) for xy in zip(entertainment_data["Long"], entertainment_data["Lat"])
    ]
    entertainment_data = gpd.GeoDataFrame(
        entertainment_data, crs=crs, geometry=entertainment_geometry
    )
    entertainment_data = entertainment_data.dropna()
    return entertainment_data


def preprocess_mbta(filename):
    mbta_data = gpd.read_file(filename)
    crs = {"init": "epsg:4326"}
    mbta_data["X"] = mbta_data["X"].astype(float)
    mbta_data["Y"] = mbta_data["Y"].astype(float)
    mbta_geometry = [Point(xy) for xy in zip(mbta_data["X"], mbta_data["Y"])]
    mbta_data = gpd.GeoDataFrame(mbta_data, crs=crs, geometry=mbta_geometry)
    mbta_data = mbta_data.dropna()
    return mbta_data


def preprocess_streetlight(filename):
    streetlight_data = gpd.read_file(filename)
    crs = {"init": "epsg:4326"}
    streetlight_data["Long"] = streetlight_data["Long"].astype(float)
    streetlight_data["Lat"] = streetlight_data["Lat"].astype(float)
    streetlight_geometry = [
        Point(xy) for xy in zip(streetlight_data["Long"], streetlight_data["Lat"])
    ]
    streetlight_data = gpd.GeoDataFrame(
        streetlight_data, crs=crs, geometry=streetlight_geometry
    )
    streetlight_data = streetlight_data.dropna()
    return streetlight_data


def preprocess_traffic_signal(filename):
    traffic_signal_data = gpd.read_file(filename)
    crs = {"init": "epsg:4326"}
    traffic_signal_data["Long"] = traffic_signal_data.geometry.x
    traffic_signal_data["Lat"] = traffic_signal_data.geometry.y
    traffic_signal_geometry = [
        Point(xy) for xy in zip(traffic_signal_data["Long"], traffic_signal_data["Lat"])
    ]
    traffic_signal_data = gpd.GeoDataFrame(
        traffic_signal_data, crs=crs, geometry=traffic_signal_geometry
    )
    traffic_signal_data = traffic_signal_data.dropna()
    return traffic_signal_data


if __name__ == "__main__":
    df = preprocess_liquor("data/liquor-licenses.csv")
    df.to_file("data/liquor_licenses_clean")
