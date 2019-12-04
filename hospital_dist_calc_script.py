from calculations.base import DistCalc
import pandas as pd
import geopandas as gpd

def main():
	feature_df = pd.read_csv("data/hospital_localtions_clean.csv")
	binned_crimes = pd.read_csv("crimes_in_bins.csv")
	calculator = DistCalc(feature_df, binned_crimes)
	calculator.calculation(["Violent Crime"])

if __name__ == "__main__":
	main()