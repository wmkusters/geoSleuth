# geoSleuth
An open-source geospatial data distribution analysis library.

## Introduction
geoSleuth is a result of a necessity for streamlined geospatial data analysis tools. Although many geospatial data science libraries exist for Python currently, geoSleuth is intended for a specific analysis over a confined area divided into subregions. A user might find this library helpful when analyzing data with geospatial locations distributed over such an area.

geoSleuth makes heavy use of the full functionality of a number of wonderful other Python libraries such as `GeoPandas` and `shapely`. This project only exists because of the massive amount of work done upfront by libraries such as those.

geoSleuth is currently at a VERY EARLY POINT IN ITS DEVELOPMENT. Full documentation is not yet available, but the library is well-documented throughout a link to the Google Doc describing in its first use case along with general library structure can be found [here](https://www.dropbox.com/s/bvz3qyyarx5qoww/6.S080%20Final%20Project%20Report.pdf?dl=0). 

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
