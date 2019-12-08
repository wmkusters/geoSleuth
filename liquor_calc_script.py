import pandas as pd
import geopandas as gpd
from calculations.base import DiscreteCalc


def main():
    feature_df = gpd.read_file("data/liquor_licenses_clean/liquor_licenses_clean.shp")
    binned_crimes = pd.read_csv("crimes_in_bins.csv")
    calculator = DiscreteCalc(feature_df, binned_crimes, feature_name="Liquor")
    subgroups = ["Violent Crime", "Theft Crime"]
    results = calculator.calculation(subgroups, group=True, to_file=True)


if __name__ == "__main__":
    main()
