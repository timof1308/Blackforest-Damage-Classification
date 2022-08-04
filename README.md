# Estimating Tree Damages Using Remote Sensing Data in the Black Forest in Germany

Masterthesis written by Timo Fischer in June 2022

## About this Repository

The design, implementation, and evaluation of a model for estimating forest condition in the Black Forest was done in the Python programming language (version 3.8). The program code was created in Jupyter Notebooks, which are presented in this directory.
The naming of the notebooks with the numbers "00" to "07" is based on the procedure of the thesis according to the CRISP-DM procedure model. The business understanding for the understanding of the forest inventories can be taken from the thesis.
- "00" to "04" covers Data Understanding as well as Data Preparation with the steps to prepare the data and connect the data for supervised learning. Each data source is examined and steps to build the attributes are shown.
- "05" covers the first step of modeling for ensemble learning
- "06" is based on the results of the previous steps and takes them further. Thus, the models are combined and evaluated
- "07" includes the application of the generated models for the estimation of a forest area in the Black Forest

At this point, the following remark should be mentioned for the content of this document: <br>.
A total of four Jupyter notebooks have been created to form the base models, which differ in target classes (healthy to dead and low to high damage). Furthermore, the data for training the models have been used unbalanced in one iteration and balanced once. Accordingly, the notebooks differ only in the target classes to be considered and the steps taken to balance the data. Further, this document includes one Jupyter notebook each for building the models with unbalanced data (for the target classes healthy to dead) and one Jupyter notbook for building the models with balanced data (for the corrupted target classes low to high).

Copy paste the following commands in order to set up your new Python Anaconda Environment:
1. Create a new Python Version 3.8 environment: `conda create -n py38thesis python=3.8`
2. Activate Anaconda Environment: `conda activate py38thesis`
3. Basis Packages I: `conda install -y -c conda-forge pandas numpy`
4. Basis Packages II: `conda install -y -c conda-forge requests beautifulsoup4 boto3 botocore dask missingno`
5. Machine Learning Packages: `conda install -y -c conda-forge scikit-learn xgboost imbalanced-learn kneed`
6. Packages for visuals: `conda install -y -c conda-forge matplotlib seaborn hvplot ipyleaflet sidecar ipyvolume geoviews`
7. Packages for remote sensing data: `conda install -y -c conda-forge geopandas xarray rioxarray planetary-computer pystac-client sat-stac intake-stac geopy shapely fiona pyproj geocube`
9. Start Jupyter Notebook: `jupyter notebook` (alternative: `jupyter lab`)

### Jupyter Notebooks
The following Jupyter notebooks are in this project:
- *"[00 - GermanyGeoLocations.ipynb](./00%20-%20GermanyGeoLocations.ipynb) "* This notebook is used to identify the geocoordinates of the Black Forest and other regions of choice. The data are taken from the Thünen Atlas.

- *"[01 - DataPrep.ipynb](./01%20-%20DataPrep.ipynb) "* This notebook is used for the data preparation of the observation data of the FVA and the Thünen Institute. Beside the preparation of the data an analysis and cleaning of the missings takes place.

- *"[02 - DEM.ipynb](./02%20-%20DEM.ipynb) "* This notebook is used to download the Digital Elevation Models and to calculate the orientation and slope of the slopes for forest areas.
- *"[02 - RemoteSensingData.ipynb](./02%20-%20RemoteSensingData.ipynb) "* This notebook is used to download remote sensing data and calculate vegetation indices for remote sensing data located in forest areas. Data are stored in raster format for months and then the maximum value per year is calculated. Data export includes vegetation indices of the whole forest area.
- *"[02 - RemoteSensingDataOuterAOI.ipynb](./02%20-%20RemoteSensingDataOuterAOI.ipynb)".ipynb* In this notebook downloading of remote sensing data and calculation of vegetation indices for observation data located outside the forest areas is performed. Data are saved in raster format for months and then the maximum value per year is calculated. Der Datenexport umfasst aufgrund der Größe lediglich die Vegetationsindizes in einem bestimmten Umkreis um die Observierungspunkte.
- *"[02 - SoilData.ipynb](./02%20-%20SoilData.ipynb)"* This notebook summarizes the soil information maps of the BGR. Different data formats (raster data vs. vector data) are combined into a uniform raster format. Missing values due to different resolutions are filled with the nearest neighbor.
- *"[02 - WeatherData.ipynb](./02%20-%20WeatherData.ipynb)"* The historical weather records of the German Weather Service are downloaded as a ZIP file and unpacked. The records of the different weather stations are combined and missing values are filled with the help of a linear interpolation per day.

- *"[03 - MergeData.ipynb](./03%20-%20MergeData.ipynb)"* for merging the labeled WZE data (which are located in the forest areas) and the formed attributes
- *"[03 - MergeDataOuterAOI.ipynb](./03%20-%20MergeDataOuterAOI.ipynb)"* For merging the labeled WZE data (which are <b>not</b> in the forest areas) and the formed attributes.

- *"[04 - LocationCluster.ipynb](./04%20-%20LocationCluster.ipynb)"* for the identification of similar geographical observation points

- The modeling for the formation of the base models consists of different steps or iterations. Accordingly, individual Jupyter notebooks are available for the modeling of the basis models:
    - For modeling, a subdivision of the data into binary classification problems is performed:
        1. damage classes are divided into the following groups:
            - healthy trees (damage class 0 : needle and leaf loss <= 10%)
            - damaged trees (damage class 1-3 : 10% > needle and leaf loss < 90%)
            - dead trees (damage class 4 : needle and leaf loss >= 90%)
        2. damaged damage classes are subdivided again without considering healthy or dead trees:
            - slightly damaged trees (damage class 1)
            - medium damaged trees (damage class 2)
            - heavy damage of the trees (damage class 3)
        3. tree species are considered on the basis of the observation data. If all tree species of a section of an observation can be assigned to conifers or deciduous trees, separate modeling is performed. If such a separation is not possible, a third modeling for mixed forests is performed, in which both deciduous trees and conifers are considered.
    - Backward feature elemination to identify 50% of the features per model
    - Hyperparameter Tuning to optimize the model parameters
    - Bagging to build the models
    - Permutation Feature Importance to check the dependency of single attributes
    - Threshhold Moving to adjust the decision boundary of the binary classification
    - Evaluation and calculation of Cohen's Kappa and F1 measure
    - Jupyter Notebooks for implementation:
        - *"[05 - EnsembleLearningBaseLearnerDamageClasses.ipynb](./05%20-%20EnsembleLearningBaseLearnerDamageClasses.ipynb)"* for building the base models for the damage classes "healthy", "damaged" and "dead" (uses the raw unbalanced data)
        - *"[05 - EnsembleLearningBaseLearnerDamageClassesBalancedData.ipynb](./05%20-%20EnsembleLearningBaseLearnerDamageClassesBalancedData.ipynb)"* for building the base models for the damage classes "healthy", "damaged" and "dead" with balanced data
        - *"[05 - EnsembleLearningBaseLearnerDamagedOnly.ipynb](./05%20-%20EnsembleLearningBaseLearnerDamagedOnly.ipynb)"* for building the base models for the damaged damage classes "low", "medium" and "high" (uses the raw unbalanced data)
        - *"[05 - EnsembleLearningBaseLearnerDamagedOnlyBalancedData.ipynb](./05%20-%20EnsembleLearningBaseLearnerDamagedOnlyBalancedData.ipynb)"* for building the base models for the damaged damage classes "low", "medium" and "strong" with balanced data

- *"[06 - EnsembleLearningMetaLearnerDamageClasses.ipynb](./06%20-%20EnsembleLearningMetaLearnerDamageClasses.ipynb)"* for forming the base models for the "healthy", "damaged" and "dead" classes into ensembles and using a weighted majority difference to determine the final estimate.
- *"[06 - EnsembleLearningMetaLearnerDamagedOnly.ipynb](./06%20-%20EnsembleLearningMetaLearnerDamagedOnly.ipynb)"* The created base models for the damaged damage classes "low", "medium" and "high" are combined into ensembles and a weighted majority difference is used to determine the final estimate.

- *"[07 - PredictBlackForest.ipynb](./07%20-%20PredictBlackForest.ipynb)"* In this notebook, an estimate of a forest area in the Black Forest is made, for which the previously determined meta-models are used.


### File Structure
The following folder structure of the data exists, in which the (raster) data, models etc. are stored:
- [./data](./data)
    - [Inventory Data](./data/ThuenenWZE) consisting of data from the Forest Research Institute of Baden Württemberg and the Thünen Institute
    - [ThünenGeoLocations](./data/ThuenenGeoLocations) with the geo-coordinates of the protected landscape areas in Germany
    - [Weather Data](./data/WeatherData)
    - [Tree distribution map](./data/ThuenenSpatialTreeData) of the Thünen Institute on the basis of the 4th Federal Forest Inventory from 2012.
    - [Digital Elevation Model](./data/DEM)
    - [Soil Data](./data/SoilData) of the Federal Institute for Geosciences and Natural Resources
    - Modeling Results:
        - [Base Models](./data/Modeling/BaseLearner)
        - [Meta Models](./data/Modeling/MetaLearner)
    - [data dump](./data/tmp) for exports within the notebooks and for transferring the data

## Medium Article
In addition, I have written an article on Medium that summarizes the results and the procedure of my Master Thesis. Click [here](https://medium.com/@timo.fischer1308/estimating-tree-damage-classes-using-remote-sensing-data-6e2d3b961f0d) for the article.
