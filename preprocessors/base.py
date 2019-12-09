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
    return liquor_data


def preprocess_entertainment(filename):
    entertainment_data = pd.read_csv(filename)
    crs = {"init": "epsg:4326"}
    entertainment_data["geometry"] = entertainment_data.Location.apply(wkt_pt_conversion)
    entertainment_data = entertainment_data.dropna(subset=["geometry"])
    entertainment_data = gpd.GeoDataFrame(entertainment_data, crs=crs, geometry=entertainment_data.geometry)
    return entertainment_data


def preprocess_mbta(filename):
    mbta_data = pd.read_csv(filename)
    crs = {"init": "epsg:4326"}
    mbta_data["Long"] = mbta_data["X"].astype(float)
    mbta_data["Lat"] = mbta_data["Y"].astype(float)
    mbta_geometry = [
        Point(xy) for xy in zip(mbta_data["Long"], mbta_data["Lat"])
    ]
    mbta_data = gpd.GeoDataFrame(mbta_data, crs=crs, geometry=mbta_geometry)
    mbta_data = mbta_data.dropna(subset=["geometry"])
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


def preprocess_charging_stations(filename):
    charging_stations_data = gpd.read_file(filename)
    crs = {"init": "epsg:4326"}
    charging_stations_data["Long"] = charging_stations_data.geometry.x
    charging_stations_data["Lat"] = charging_stations_data.geometry.y
    charging_stations_geometry = [
        Point(xy) for xy in zip(charging_stations_data["Long"], charging_stations_data["Lat"])
    ]
    charging_stations_data = gpd.GeoDataFrame(
        charging_stations_data, crs=crs, geometry=charging_stations_geometry
    )
    charging_stations_data = charging_stations_data.dropna()
    return charging_stations_data


def preprocess_colleges(filename):
    college_data = gpd.read_file(filename)
    crs = {"init": "epsg:4326"}
    college_data["Long"] = college_data.geometry.x
    college_data["Lat"] = college_data.geometry.y
    college_geometry = [
        Point(xy) for xy in zip(college_data["Long"], college_data["Lat"])
    ]
    college_data = gpd.GeoDataFrame(
        college_data, crs=crs, geometry=college_geometry
    )
    college_data = college_data.dropna()
    return college_data


def preprocess_pools(filename):
    pool_data = gpd.read_file(filename)
    crs = {"init": "epsg:4326"}
    pool_data["Long"] = pool_data.geometry.x
    pool_data["Lat"] = pool_data.geometry.y
    pool_geometry = [
        Point(xy) for xy in zip(pool_data["Long"], pool_data["Lat"])
    ]
    pool_data = gpd.GeoDataFrame(
        pool_data, crs=crs, geometry=pool_geometry
    )
    pool_data = pool_data.dropna()
    return pool_data


def preprocess_community_centers(filename):
    comm_center_data = gpd.read_file(filename)
    crs = {"init": "epsg:4326"}
    comm_center_data["Long"] = comm_center_data.geometry.x
    comm_center_data["Lat"] = comm_center_data.geometry.y
    comm_center_geometry = [
        Point(xy) for xy in zip(comm_center_data["Long"], comm_center_data["Lat"])
    ]
    comm_center_data = gpd.GeoDataFrame(
        comm_center_data, crs=crs, geometry=comm_center_geometry
    )
    comm_center_data = comm_center_data.dropna()
    return comm_center_data


def preprocess_public_schools(filename):
    public_school_data = gpd.read_file(filename)
    crs = {"init": "epsg:4326"}
    public_school_data["Long"] = public_school_data.geometry.x
    public_school_data["Lat"] = public_school_data.geometry.y
    public_school_geometry = [
        Point(xy) for xy in zip(public_school_data["Long"], public_school_data["Lat"])
    ]
    public_school_data = gpd.GeoDataFrame(
        public_school_data, crs=crs, geometry=public_school_geometry
    )
    public_school_data = public_school_data.dropna()
    return public_school_data


def preprocess_private_schools(filename):
    private_school_data = gpd.read_file(filename)
    crs = {"init": "epsg:4326"}
    private_school_data["Long"] = private_school_data.geometry.x
    private_school_data["Lat"] = private_school_data.geometry.y
    private_school_geometry = [
        Point(xy) for xy in zip(private_school_data["Long"], private_school_data["Lat"])
    ]
    private_school_data = gpd.GeoDataFrame(
        private_school_data, crs=crs, geometry=private_school_geometry
    )
    private_school_data = private_school_data.dropna()
    return private_school_data


def preprocess_trees(filename):
    tree_data = gpd.read_file(filename)
    crs = {"init": "epsg:4326"}
    tree_data["Long"] = tree_data.geometry.x
    tree_data["Lat"] = tree_data.geometry.y
    tree_geometry = [
        Point(xy) for xy in zip(tree_data["Long"], tree_data["Lat"])
    ]
    tree_data = gpd.GeoDataFrame(
        tree_data, crs=crs, geometry=tree_geometry
    )
    tree_data = tree_data.dropna()
    return tree_data


def preprocess_police_stations(filename):
    station_data = gpd.read_file(filename)
    crs = {"init": "epsg:4326"}
    station_data["Long"] = station_data.geometry.x
    station_data["Lat"] = station_data.geometry.y
    station_geometry = [
        Point(xy) for xy in zip(station_data["Long"], station_data["Lat"])
    ]
    station_data = gpd.GeoDataFrame(
        station_data, crs=crs, geometry=station_geometry
    )
    station_data = station_data.dropna()
    return station_data



if __name__ == "__main__":
    df = preprocess_liquor("data/liquor-licenses.csv")
    df.to_file("data/liquor_licenses_clean")
