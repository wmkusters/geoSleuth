from calculations.base import DistCalc
import pandas as pd

def main():
	binned_crimes = pd.read_csv("crimes_in_bins.csv")
	calculator = DistCalc()

if __name__ == "__main__":