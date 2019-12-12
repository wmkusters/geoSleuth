# geoSleuth
An open-source geospatial data distribution analysis library.

## Introduction
geoSleuth is a result of a necessity for streamlined geospatial data analysis tools. Although many geospatial data science libraries exist for Python currently, geoSleuth is intended for a specific analysis over a confined area divided into subregions. A user might find this library helpful when analyzing data with geospatial locations distributed over such an area.

geoSleuth makes heavy use of the full functionality of a number of wonderful other Python libraries such as `GeoPandas` and `shapely`. This project only exists because of the massive amount of work done upfront by libraries such as those.

geoSleuth is currently at a VERY EARLY POINT IN ITS DEVELOPMENT. Full documentation is not yet available, but the library is well-documented throughout and a link to the Google Doc describing the library's creation and first use case  can be found [here](https://www.dropbox.com/s/bvz3qyyarx5qoww/6.S080%20Final%20Project%20Report.pdf?dl=0). 

## Installation
geoSleuth requires a number of Python libraries, with explicit imports enumerated here:

* `matplotlib`
* `pandas`
* `numpy`
* `geopandas`
* `shapely`
* `sklearn`
* `xgboost`


In addition to these requirements, geopandas requires some more complicated packages:

* `fiona`
* `descartes`
* `rtree`


Due to the fact that some of these modules depend on C libraries, a Conda install is recommended for `geopandas`. Full installation documentation for `geopandas` is linked [here](https://geopandas.readthedocs.io/en/latest/install.html).

## Usage
geoSleuth requires these three things to be of any use: some constrained geospatial area divided into subregions and able to read into a `GeoDataFrame`, a dataset of interest that falls geospatially over this constrained area, and a feature dataset. All three of these items should be able to be represented as `GeoDataFrames` in that they need to have some `geometry` column that has a value for every row. The distribution dataset is spatially joined to the region dataset conditioned on a subregion containing some point in the distribution, resulting in a `GeoDataFrame` with every point of the distribution of interest joined to the subregion it falls within. This computation is expected to be expensive, and as such is done before any calculation or analysis and is intended to be run once and saved, to be read back into a `GeoDataFrame` while the library is being used.

Once this region assigned data table is created, geoSleuth is used to investigate the relationship of various other geospatial features (falling across the same region) with regards to the distribution of interest. In the first use case, the example was a spatial partioning of Boston, the distribution of crime incidents falling across these partitions, and a variety of features to be used for investigation.

## Demo
To walk through a single case usage, run the bash file `example.sh` to start Jupyter Notebook for a walkthrough.
