import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt
import numpy as np

class BaseProcessor:
    def __init__(self, filename):
        # check if csv or shp
        self.crs = {"init": "epsg:4326"}
        if filename.endswith('shp'):
            self.data = gpd.read_file(filename)
        elif filename.endswith('csv'):
            self.data = pd.read_csv(filename)
            # more processing happens in the child class (only relevant for csv)
        else:
            raise OSError("Processor only accepts .csv and .shp files")

    def process(self):
        # cast as GeoDataFrame and drop empty geometry
        # at this point, self.data should have a geometry column
        # (either from original shp file or created in specific processor)
        self.data = gpd.GeoDataFrame(self.data, crs=self.crs, geometry=self.data.geometry)
        self.data = self.data[~(self.data.geometry.is_empty | self.data.geometry.isna())]
        return self.data

    @staticmethod
    def wkt_pt_conversion(pt_string):
        """
        parameters:
            pt_string: string describing a point to be converted
                       to well-known text format currently just
                       converts points enclosed in () with a comma
                       and need to have the Lat/Long parsed into
                       x/y.
        returns
            Null value in the case of a point == (0, 0), or the
            shapely POINT object of the wkt point.
        """
        pt_string = pt_string.replace("(", "")
        pt_string = pt_string.replace(")", "")
        pt_string = pt_string.replace(" ", "")
        pt_string = pt_string.split(",")
        pt_string = "(" + pt_string[1] + " " + pt_string[0] + ")"
        to_wkt = wkt.loads("POINT " + pt_string)
        if to_wkt.x == 0.0 or to_wkt.y == 0.0:
            return np.nan
        return to_wkt


class HospitalProcessor(BaseProcessor):
    def __init__(self, filename):
        BaseProcessor.__init__(self, filename)

        # dataset-specific processing
        self.data['Long'] = self.data.XCOORD.apply(lambda x: x / -10e5)
        self.data['Lat'] = self.data.YCOORD.apply(lambda y: y / 10e5)
        self.data['geometry'] = [Point(xy) for xy in zip(self.data['Long'], self.data['Lat'])]


class LiquorProcessor(BaseProcessor):
    def __init__(self, filename):
        BaseProcessor.__init__(self, filename)

        # dataset-specific processing
        self.data = self.data.dropna(subset=['Location'])
        self.data['geometry'] = self.data.Location.apply(self.wkt_pt_conversion)


class EntertainmentProcessor(BaseProcessor):
    def __init__(self, filename):
        BaseProcessor.__init__(self, filename)

        # dataset-specific processing
        self.data = self.data.dropna(subset=['Location'])
        self.data['geometry'] = self.data.Location.apply(self.wkt_pt_conversion)


class MbtaProcessor(BaseProcessor):
    def __init__(self, filename):
        BaseProcessor.__init__(self, filename)

        # dataset-specific processing
        self.data['Long'] = self.data["X"].astype(float)
        self.data['Lat'] = self.data["Y"].astype(float)
        self.data['geometry'] = [Point(xy) for xy in zip(self.data['Long'], self.data['Lat'])]


class StreetlightProcessor(BaseProcessor):
    def __init__(self, filename):
        BaseProcessor.__init__(self, filename)

        # dataset-specific processing
        self.data["Long"] = self.data["Long"].astype(float)
        self.data["Lat"] = self.data["Lat"].astype(float)
        self.data['geometry'] = [Point(xy) for xy in zip(self.data['Long'], self.data['Lat'])]

