from calculations.base import DistCalc
import pandas as pd
import geopandas as gpd


def main():
    feature_df = pd.read_csv("data/hospital_localtions_clean.csv")
    binned_crimes = pd.read_csv("crimes_in_bins.csv")
    calculator = DistCalc(feature_df, binned_crimes, feature_name="Hospital")
    subgroups = ["Violent Crime", "Theft Crime"]
    results = calculator.calculation(subgroups, feature_function=min, group=True, to_file=True)
    


if __name__ == "__main__":
    main()
