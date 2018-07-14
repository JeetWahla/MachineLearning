# MachineLearning
LinearRegression_usecase
I am using pysparkML and sckitlearn for machine learning use case 
HOUSEING PRICE PREDICTION
===========================
Assignment Description
 
Data Taken From Kaggle 
 
Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
 
With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this assignment predict the final price of each home.
 
 
Skills:-
Creative feature engineering 
Pyspark
Advanced regression techniques like random forest and gradient boosting
 
Acknowledgments
The Ames Housing dataset was compiled by Dean De Cock for use in data science education. It's an incredible alternative for data scientists looking for a modernized and expanded version of the often cited Boston Housing dataset.
 
 
Data fields
Here's a brief version of what you'll find in the data description file.
 
SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
MSSubClass: The building class
MSZoning: The general zoning classification
LotFrontage: Linear feet of street connected to property
LotArea: Lot size in square feet
Street: Type of road access
Alley: Type of alley access
LotShape: General shape of property
LandContour: Flatness of the property
Utilities: Type of utilities available
LotConfig: Lot configuration
LandSlope: Slope of property
Neighborhood: Physical locations within Ames city limits
Condition1: Proximity to main road or railroad
Condition2: Proximity to main road or railroad (if a second is present)
BldgType: Type of dwelling
HouseStyle: Style of dwelling
OverallQual: Overall material and finish quality
OverallCond: Overall condition rating
YearBuilt: Original construction date
YearRemodAdd: Remodel date
RoofStyle: Type of roof
RoofMatl: Roof material
Exterior1st: Exterior covering on house
Exterior2nd: Exterior covering on house (if more than one material)
MasVnrType: Masonry veneer type
MasVnrArea: Masonry veneer area in square feet
ExterQual: Exterior material quality
ExterCond: Present condition of the material on the exterior
Foundation: Type of foundation
BsmtQual: Height of the basement
BsmtCond: General condition of the basement
BsmtExposure: Walkout or garden level basement walls
BsmtFinType1: Quality of basement finished area
BsmtFinSF1: Type 1 finished square feet
BsmtFinType2: Quality of second finished area (if present)
BsmtFinSF2: Type 2 finished square feet
BsmtUnfSF: Unfinished square feet of basement area
TotalBsmtSF: Total square feet of basement area
Heating: Type of heating
HeatingQC: Heating quality and condition
CentralAir: Central air conditioning
Electrical: Electrical system
1stFlrSF: First Floor square feet
2ndFlrSF: Second floor square feet
LowQualFinSF: Low quality finished square feet (all floors)
GrLivArea: Above grade (ground) living area square feet
BsmtFullBath: Basement full bathrooms
BsmtHalfBath: Basement half bathrooms
FullBath: Full bathrooms above grade
HalfBath: Half baths above grade
Bedroom: Number of bedrooms above basement level
Kitchen: Number of kitchens
KitchenQual: Kitchen quality
TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
Functional: Home functionality rating
Fireplaces: Number of fireplaces
FireplaceQu: Fireplace quality
GarageType: Garage location
GarageYrBlt: Year garage was built
GarageFinish: Interior finish of the garage
GarageCars: Size of garage in car capacity
GarageArea: Size of garage in square feet
GarageQual: Garage quality
GarageCond: Garage condition
PavedDrive: Paved driveway
WoodDeckSF: Wood deck area in square feet
OpenPorchSF: Open porch area in square feet
EnclosedPorch: Enclosed porch area in square feet
3SsnPorch: Three season porch area in square feet
ScreenPorch: Screen porch area in square feet
PoolArea: Pool area in square feet
PoolQC: Pool quality
Fence: Fence quality
MiscFeature: Miscellaneous feature not covered in other categories
MiscVal: $Value of miscellaneous feature
MoSold: Month Sold
YrSold: Year Sold
SaleType: Type of sale
SaleCondition: Condition of sale
 
 
From Starting first we have to remove header from data so now data header already removed in dataset
 
Mention the path of your data file
we will read the data as sc.textFile(path)
 
1st we will count the data and then split the data from comma seperated 
 
mention catagorical column for mapping 
type_columns=[2,5,7,8,9,10,11,12,13,14,15,16,21,22,23,24,27,28,29,39,40,41,53,55,65,78,79]
 
again mention categorical columns having null values
type_columns_with_NA=[6,25,30,31,32,33,35,42,57,58,60,63,64,72,73,74]
 
mention target value saleprice_column=80
 
prepare data for modify in label and feature by use of LabeledPoint
and map with two different function created in programm
extract_features_dt,extract_features
 
now here all data has prepared now we use linear regression with LinearRegressionWithSGD
 
we got the true_vs_predicted value and
we got
Linear Model predictions: [(208500.0, -1.3111060925180484e+75), (181500.0, -1.4720767452081686e+75), (223500.0, -1.7050281430818638e+75), (140000.0, -1.4631365187530982e+75), (250000.0, -2.1369709269890862e+75)]
 
which is seems that our prediction are not accurate now 
 
find the Linear Model - Mean Squared Error: 
 
45192838358763 more than that 
 
now we total see this model is not able to predict our values
 
even we used true_vs_predicted_log or Root Mean Squared Log Error
 
still our algorithum not able to predict inspite of using tunning performance on 
house data
 
so we move to Decision Tree 
 
now again using feature vector for decision tree
 
our result come like below:
 
Decision Tree predictions: [(208500.0, 190334.33561643836), (181500.0, 147907.61375661375), (223500.0, 190334.33561643836), (140000.0, 156058.38888888888), (250000.0, 307760.1111111111)]
Decision Tree depth: 5
Decision Tree number of nodes: 63
 
its showing pretty good reuslt now we check our accuracy
 
but still not giving us good accuracy
Root Mean Squared Log Error: 0.1736
its like may be our data under fitting
 
try taking log value
still not good accuracy 
Root Mean Squared Log Error: 0.1610
 
 
now we use gradient boosted regression
 
with this algo 
 
we get optimized accuracy of
Root Mean Squared Log Error: 0.2214
 
===================================================================================
 
Cycle Rental
 
information on the dataset, including the variable names and descriptions. Take a look at the file, and you will see that we have the following variables available:
 
instant: This is the record ID
 
dteday: This is the raw date
 
season: This is different seasons such as spring, summer, winter, and fall
 
yr: This is the year (2011 or 2012)
mnth: This is the month of the year
 
hr: This is the hour of the day
holiday: This is whether the day was a holiday or not
 
weekday: This is the day of the week
 
workingday: This is whether the day was a working day or not
 
weathersit: This is a categorical variable that describes the weather at a particular time
 
temp: This is the normalized temperature
 
atemp: This is the normalized apparent temperature
 
hum: This is the normalized humidity
 
windspeed: This is the normalized wind speed
 
cnt: This is the target variable, that is, the count of bike rentals for that hour
 
========
 
we have 17,379 hourly records in our dataset. We have inspected the column names already. We will ignore the record ID and raw date columns. We will also ignore the casual and registered count target variables and focus on the overall count variable, cnt(which is the sum of the other two counts). We are left with 12 variables. The first eight are categorical, while the last 4 are normalized real-valued variables.
 
 We now have the mappings for each variable, and we can see how many values in total we need for our binary vector representation:
 
Feature vector length for categorical features: 57
Feature vector length for numerical features: 4
Total feature vector length: 61
Training a regression model on the bike sharing dataset
 
We're ready to use the features we have extracted to train our models on the bike sharing data. First, we'll train the linear regression model and take a look at the first few predictions that the model makes on the data:
 
linear_model = LinearRegressionWithSGD.train(data, iterations=10, step=0.1, intercept=False)
 
so we get some prediction of our model 
Linear Model predictions: [(16.0, 117.89250386724846), (40.0, 116.2249612319211), (32.0, 116.02369145779235), (13.0, 115.67088016754433), (1.0, 115.56315650834317)]
 
now we have to find root mean squared error
Mean Squared Error: 50685.5559
Mean Absolue Error: 155.2955
Root Mean Squared Log Error: 1.5411
 
=========
Creating feature vectors for the decision tree:
Decision Tree feature vector: [1.0,0.0,1.0,0.0,0.0,6.0,0.0,1.0,0.24,0.2879,0.81,0.0]
Decision Tree feature vector length: 12
 
using decision tree algo DecisionTree
 
Decision Tree predictions: [(16.0, 54.913223140495866), (40.0, 54.913223140495866), (32.0, 53.171052631578945), (13.0, 14.284023668639053), (1.0, 14.284023668639053)]
Decision Tree depth: 5
Decision Tree number of nodes: 63
 
achieved good accuracy for our dataset
 
log - Mean Squared Error: 11611.4860
log - Mean Absolue Error: 71.1502
Root Mean Squared Log Error: 0.6251
accuracy 62%
 
and with our log data we optimized our accuracy by 2%
 
log - Mean Squared Error: 14781.5760
log - Mean Absolue Error: 76.4131
Root Mean Squared Log Error: 0.6406
Non log-transformed predictions:
[(16.0, 54.913223140495866), (40.0, 54.913223140495866), (32.0, 53.171052631578945)]
 
accuracy now 64%
====================
Gradient BOOSTED
 
Gradient BOOSTED predictions: [(32.0, 21.043168257608315), (1.0, 21.043168257608315), (2.0, 20.78800044083532), (14.0, 122.94779932705194), (36.0, 122.94779932705194)]
 
from this algo we achieved good accuracy 
5274
log - Mean Squared Error: 13350.5537
log - Mean Absolue Error: 80.6329
Root Mean Squared Log Error: 0.7853
accuracy 78%
