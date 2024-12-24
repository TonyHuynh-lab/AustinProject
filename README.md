# AustinProject
<ins>**Inspiration**</ins></br>
As a resident here at UT Austin, I have come across countless CapMetro bikes around my campus. 
Out of curiosity, I have wondered how often these bikes are being used. This question led
me to factors that may influence bike usage, such as the time of the day and whether it is the weekend
or not. Predicting the demand for bikes at a station during a specific time could provide valuable
insight into the allocation of these bikes. Using what I have learned through self-studying of machine
learning techniques, I sought to develop a model that could accurately provide such predictions.</br></br>
<ins>**Required CSV Files**</ins></br>
*austin-bikeshare_trips.csv* </br>
*austin_bikeshare_stations.csv* </br>
These files consist of data on >649,000 bike trips over 2013-2017, which is generously provided by the City of Austin. </br>
More information/download can be found here: https://www.kaggle.com/datasets/jboysen/austin-bike/data </br></br>
<ins>**What Are The Files?**</ins></br>
*preprocessing.py* </br>
This script is essential for filtering and preprocessing bike trip and station data to prepare it for modeling. 
It extracts key features such as hour, day_of_week, start_station_id (the previous three features are
grouped to calculate the num_trips for each combination), and the geographical locations and statuses 
of stations. These features are contained in a singular data frame (station_usage) for further analysis. 
Additionally, the script introduces a lag feature (num_trips_lag) that captures temporal trends in the number of trips 
at each station. station_usage contains 10010 indices after filtering out missing data and merging the two CSV files.
