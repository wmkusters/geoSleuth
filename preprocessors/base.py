import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def preprocess_hospital(filename):
    hospital_data = pd.read_csv(filename)
    crs = {"init": "epsg:4326"}
    hospital_data.XCOORD = hospital_data.XCOORD.apply(lambda x: x / -10e5)
    hospital_data.YCOORD = hospital_data.YCOORD.apply(lambda y: y / 10e5)
    geometry = [Point(xy) for xy in zip(hospital_data.XCOORD, hospital_data.YCOORD)]
    hospital_data = gpd.GeoDataFrame(hospital_data, crs=crs, geometry=geometry)
    hospital_data = hospital_data.dropna()
    return hospital_data


def preprocess_liquor(filename):
    liquor_data = gpd.read_file(filename)
    liquor_data["Long"] = liquor_data["Long"].astype(float)
    liquor_data["Lat"] = liquor_data["Lat"].astype(float)
    liquor_geometry = [Point(xy) for xy in zip(liquor_data["Long"], liquor_data["Lat"])]
    liquor_data = gpd.GeoDataFrame(liquor_data, crs=crs, geometry=liquor_geometry)


if __name__ == "__main__":
    df = preprocess_hospital("data/hospital-locations.csv")
    df.to_file("data/hospital_localtions_clean.shp", driver="ESRI Shapefile")
