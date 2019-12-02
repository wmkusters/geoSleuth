import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def subgroup_gen():
    # get + format crime data
    crime_data = pd.read_csv("../data/boston_crime.csv")
    crs = {"init": "epsg:4326"}
    geometry = [Point(xy) for xy in zip(crime_data["Long"], crime_data["Lat"])]
    crime_data = gpd.GeoDataFrame(crime_data, crs=crs, geometry=geometry)

    # clean crime data of crimes to not include
    non_crimes = ["Medical Assistance", "Fire Related Reports", "Biological Threat"]
    recoveries = [
        "Property Found",
        "Recovered Stolen Property",
        "Auto Theft Recovery",
        "Missing Person Located",
    ]
    interpersonal = [
        "Phone Call Complaints",
        "Restraining Order Violations",
        "Offenses Against Child / Family",
        "Landlord/Tenant Disputes",
    ]
    geo_specific = [
        "Harbor Related Incidents",
        "Prisoner Related Incidents",
        "Evading Fare",
        "Aircraft",
    ]
    no_include = non_crimes + recoveries + interpersonal + geo_specific
    crime_data = crime_data[~crime_data["OFFENSE_CODE_GROUP"].isin(no_include)]

    # filter for subgroupings
    car_parking_incidents = [
        "License Plate Related Incidents",
        "License Violation",
        "Larceny From Motor Vehicle",
        "Towed",
        "Auto Theft",
    ]
    car_on_road_incidents = [
        "License Plate Related Incidents",
        "License Violation",
        "Operating Under the Influence",
        "Motor Vehicle Accident Response",
    ]
    all_car_incidents = car_parking_incidents + car_on_road_incidents

    physical_violent_crimes = [
        "Harassment",
        "Criminal Harassment",
        "Simple Assault",
        "Aggravated Assault",
        "Homicide",
        "Manslaughter",
        "Explosives",
        "Arson",
        "Ballistics",
        "Firearm Violations",
        "Firearm Discovery",
        "HUMAN TRAFFICKING",
        "HUMAN TRAFFICKING - INVOLUNTARY SERVITUDE",
    ]
    all_theft_crime = [
        "Commercial Burglary",
        "Robbery",
        "Larceny From Motor Vehicle",
        "Residential Burglary",
        "Property Lost",
        "Auto Theft",
        "Larceny",
        "Other Burglary",
    ]

    # crimes not theft, violent, or on road
    financial_crimes = [
        "Counterfeiting",
        "Gambling",
        "Confidence Games",
        "Fraud",
        "Embezzlement",
    ]
    other_non_violent = [
        "Assembly or Gathering Violations",
        "Liquor Violation",
        "Disorderly Conduct",
        "Vandalism",
        "Property Related Damage",
        "Drug Violation",
        "Verbal Disputes",
        "Prostitution",
    ]
    other_incidents = [
        "Service",
        "Police Service Incidents",
        "Warrant Arrests",
        "Search Warrants",
        "Violations",
        "Other",
        "Burglary - No Property Taken",
        "HOME INVASION",
    ]
    other_crimes_definite = financial_crimes + other_non_violent + other_incidents
    all_crimes_definite = (
        all_car_incidents
        + physical_violent_crimes
        + all_theft_crime
        + other_crimes_definite
    )

    non_definite_crimes = [
        "Investigate Person",
        "Bomb Hoax",
        "Investigate Property",
        "INVESTIGATE PERSON",
        "Missing Person Reported",
    ]
    other_crimes_indefinite = other_crimes_definite + non_definite_crimes
    all_crimes_indefinite = all_crimes_definite + non_definite_crimes

    crime_subgroupings = {
        "Violent Crime": physical_violent_crimes,
        "Theft Crime": all_theft_crime,
        "On Road Crime": car_on_road_incidents,
        "Other Definite Crimes": other_crimes_definite,
        "Other Non-Definite Crimes": other_crimes_indefinite,
        "All Definite Crimes": all_crimes_definite,
        "All Non-Definite Crimes": all_crimes_indefinite,
    }
    return crime_subgroupings

subgroups = subgroup_gen()
print(subgroups)
