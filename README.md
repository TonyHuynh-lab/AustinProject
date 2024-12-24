# AustinProject
<ins>**Inspiration**</ins></br>
  As a resident here at UT Austin, I have come across countless CapMetro bikes around my campus. 
Out of curiosity, I have wondered how often these bikes are being used. This question led
me to factors that may influence bike usage, such as the time of the day and whether it is the weekend
or not. Predicting the demand for bikes at a station during a specific time could provide valuable
insight into the allocation of these bikes. Using what I have learned through self-studying of machine
learning techniques, I sought to develop a model that could accurately provide such predictions.</br></br>
<ins>**Goal**</ins></br>
  To predict bike demand at different stations at a given time. </br></br>
<ins>**Required CSV Files**</ins></br>
*austin-bikeshare_trips.csv* </br>
*austin_bikeshare_stations.csv* </br>
  These files consist of data on >649,000 bike trips over 2013-2017, which is generously provided by the City of Austin. </br>
More information/download can be found here: https://www.kaggle.com/datasets/jboysen/austin-bike/data </br></br>
<ins>**What Are The Files?**</ins></br>
***preprocessing.py*** </br>
  This script is essential for filtering and preprocessing bike trip and station data to prepare it for modeling. 
It extracts key features such as hour, day_of_week, start_station_id (the previous three features are
grouped to calculate the num_trips for each combination), and the geographical locations and statuses 
of stations. These features are contained in a singular data frame (station_usage) for further analysis. 
Additionally, the script introduces a lag feature (num_trips_lag) that captures temporal trends in the number of trips 
at each station. station_usage contains 10010 indices after filtering out missing data and merging the two CSV files. </br></br>
***model.py*** </br>
  This script focuses on building, training, and evaluating two machine-learning models-Random Forest and XGBoost. 
The models are trained on data preprocessed from *preprocessing.py* and use an 80-20 data split for the training
and test set, respectively. The performance of these models is evaluated by metrics such as mean absolute error
and root mean squared error. After performing hyperparameter optimization on both models with randomly sampled
combinations, it was clear that XGBoost would be the desired learning model for this experiment. </br>
* Random Forest (Tuned) - MAE: 18.45, RMSE: 33.88 </br>
* XGBoost (Tuned) - MAE: 16.29, RMSE: 28.81 </br>

The following diagram represents the prediction of the number of bike trips by the XGBoost model with respect to the actual
number of bike trips from station_usage. </br>
<img src="https://github.com/TonyHuynh-lab/AustinProject/blob/main/XGBoostPrediction.png?raw=true" alt="alt text" width="900"/></br></br>
***user.py***

