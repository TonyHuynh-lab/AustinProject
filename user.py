import pandas as pd
import joblib
from preprocessing import preprocess

# Load model
model = joblib.load('model.pkl')

def preprocess_and_predict(station_id, hour, day_of_week):
    # File paths
    trips_file = 'austin_bikeshare_trips.csv'
    stations_file = 'austin_bikeshare_stations.csv'

    # Preprocess the data
    station_usage = preprocess(trips_file, stations_file)

    station_info = station_usage[(station_usage['start_station_id'] == station_id) & 
                                 (station_usage['hour'] == hour) & 
                                 (station_usage['day_of_week'] == day_of_week)]
    
    # Check if the station ID is valid
    if not station_info.empty:
        latitude = station_info['latitude'].values[0]
        longitude = station_info['longitude'].values[0]
        status = station_info['status'].values[0]
        num_trips_lag = station_info['num_trips_lag'].values[0]
    else:
        raise ValueError("The station ID provided is not valid.")

    # Prepare the input data for prediction
    input_df = pd.DataFrame({
        'hour': [hour],
        'day_of_week': [day_of_week],
        'latitude': [latitude],
        'longitude': [longitude],
        'status': [status],
        'num_trips_lag': [num_trips_lag]
    })

    # Expected Value
    actual_bikes = station_info['num_trips'].values[0]

    print(f"Actual number of trips at station {station_id} at hour {hour}:00 on day {day_of_week}: "
        f"{actual_bikes}")
    
    # Actual Value Prediction
    prediction = model.predict(input_df)
    return prediction[0]

# Get user input
try:
    station_id = int(input("Enter station ID: "))
    hour = int(input("Enter hour (0-23): "))
    day_of_week = int(input("Enter day of the week (0-6, where 0 = Monday and 6 = Sunday): "))

    # Check preconditions
    if not (0 <= hour <= 23):
        raise ValueError("Hour must be between 0 and 23.")
    if not (0 <= day_of_week <= 6):
        raise ValueError("Day of the week must be between 0 and 6.")
    
    # Call the function and display the result
    prediction = preprocess_and_predict(station_id, hour, day_of_week)
    print(f"The predicted number of trips is {prediction:.2f}.")
    
except ValueError as e:
    print(f"Error: {e}")