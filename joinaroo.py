from vars.subgroupings import subgroups
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt


def main():
    census_tracts = gpd.read_file("data/Census_2010_Tracts")
    census_tracts["orig_area"] = census_tracts["geometry"].area
    census_tracts = pd.DataFrame(census_tracts[["OBJECTID", "orig_area"]])

    crime_data = pd.read_csv("data/boston_crime.csv")
    crs = {"init": "epsg:4326"}
    crime_data = gpd.GeoDataFrame(
        crime_data,
        crs=crs,
        geometry=gpd.points_from_xy(crime_data.Long, crime_data.Lat),
    )
    crime_data["crime_point"] = crime_data["geometry"]

    bin_data = gpd.read_file("data/Census_Bins/Census_Bins.shp")
    bin_data.index.name = "bin_id"
    bin_data = bin_data.reset_index()
    bin_data = bin_data[["bin_id", "OBJECTID", "geometry"]]
    bin_data = bin_data.rename(columns={"OBJECTID": "OBJECTID_l"})

    census_tracts = census_tracts.rename(columns={"OBJECTID": "OBJECTID_r"})
    bin_data = bin_data.merge(
        census_tracts,
        left_on="OBJECTID_l",
        right_on="OBJECTID_r",
        how="inner",
        validate="many_to_one",
    )
    bin_data["area_proportion"] = bin_data.geometry.area / bin_data.orig_area
    bin_data = bin_data[["bin_id", "OBJECTID_l", "area_proportion", "geometry"]].rename(
        columns={"OBJECTID_l": "OBJECTID"}
    )
    crime_data = crime_data[
        [
            "INCIDENT_NUMBER",
            "OFFENSE_CODE",
            "OFFENSE_CODE_GROUP",
            "DISTRICT",
            "SHOOTING",
            "OCCURRED_ON_DATE",
            "geometry",
            "crime_point",
        ]
    ]
    bins_to_crime = gpd.sjoin(bin_data, crime_data, op="contains")
    bins_to_crime["centroid"] = bins_to_crime.centroid
    bins_to_crime.to_csv("crimes_in_bins.csv")


if __name__ == "__main__":
    main()
