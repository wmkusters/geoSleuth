import pandas as pd
import geopandas as gpd
from calculations.base import DiscreteCalc


def main():
    feature_df = gpd.read_file("data/liquor_licenses_clean/liquor_licenses_clean.shp")
    binned_crimes = pd.read_csv("crimes_in_bins.csv")
    calculator = DiscreteCalc(feature_df, binned_crimes)
    subgroups = ["Violent Crime", "Theft Crime"]
    results = calculator.calculation(subgroups)
    for result, subgroup in zip(results, subgroups):
        # print(result[["bin_id", "OBJECTID", "OCCURRED_ON_DATE", "OFFENSE_CODE_GROUP", "crime_point"]].head())
        result.to_csv(
            "Liquor_Results/" + subgroup.replace(" ", "") + ".csv"
        )


if __name__ == "__main__":
    main()
