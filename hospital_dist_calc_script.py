from calculations.base import DistCalc
from preprocessors.base import HospitalProcessor
from analysis.base import BaseAnalyzer
import pandas as pd
import geopandas as gpd


def inverse_square_sum(distances):
    result = 0
    for distance in distances:
        result += 1 / (distance ** 2)
    if result > 15:
        result = 15
    return result


def main():
    feature_df = HospitalProcessor("data/hospital-locations.csv").data
    binned_crimes = pd.read_csv("data/crimes_in_bins.csv")
    calculator = DistCalc(feature_df, binned_crimes, feature_name="Hospital_InvSq")
    subgroups = ["Violent Crime", "Theft Crime"]
    results = calculator.calculation(
        subgroups, feature_function=inverse_square_sum, group=True, to_file=False
    )
    # print(results["Violent Crime"][results["Violent Crime"].feature > 15])
    # raise SystemError(0)
    analyzer = BaseAnalyzer(results["Violent Crime"])
    analyzer.linear_model(plot=True)


if __name__ == "__main__":
    main()
