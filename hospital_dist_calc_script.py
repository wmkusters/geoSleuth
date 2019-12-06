from calculations.base import DistCalc
import pandas as pd
import geopandas as gpd


def main():
    feature_df = pd.read_csv("data/hospital_locations_clean.csv")
    binned_crimes = pd.read_csv("crimes_in_bins.csv")
    calculator = DistCalc(feature_df, binned_crimes)
    subgroups = ["Violent Crime", "Theft Crime"]
    results = calculator.calculation(subgroups, min)
    for result, subgroup in zip(results, subgroups):
        # print(result[["bin_id", "OBJECTID", "OCCURRED_ON_DATE", "OFFENSE_CODE_GROUP", "crime_point"]].head())
        result.to_csv(
            "Hospital_Results/" + subgroup.replace(" ", "") + "_min.csv"
        )


if __name__ == "__main__":
    main()
