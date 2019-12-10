from calculations.base import DistCalc
from preprocessors.base import preprocess_hospital
from analysis.base import BaseAnalyzer
import pandas as pd
import geopandas as gpd

def inverse_square_sum(distances):
    result = 0
    for distance in distances:
        result += 1/(distance*2)
    return result

def main():
    feature_df = preprocess_hospital("data/hospital-locations.csv")
    binned_crimes = pd.read_csv("data/crimes_in_bins.csv")
    calculator = DistCalc(feature_df, binned_crimes, feature_name="Hospital_InvSq")
    subgroups = ["Violent Crime", "Theft Crime"]
    results = calculator.calculation(subgroups, feature_function=inverse_square_sum, group=True, to_file=False)
    print(results.head())
    raise SystemError(0)
    analyzer = BaseAnalyzer(results)
    BaseAnalyzer.linear_model(plot=True)


if __name__ == "__main__":
    main()
