import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import randint
from preprocessing import preprocess

# File paths
trips_file = 'austin_bikeshare_trips.csv'
stations_file = 'austin_bikeshare_stations.csv'

# Preprocess the data
station_usage = preprocess(trips_file, stations_file)

#print(station_usage.head())
#print(station_usage.describe())

# Ensure 'start_station_id', 'hour', and 'day_of_week' are included for comparison
X_full = station_usage[['start_station_id', 'hour', 'day_of_week', 'latitude', 'longitude', 
                        'status', 'num_trips_lag']]

# Split the data into features (X) and target (y)
X = X_full[['hour', 'day_of_week', 'latitude', 'longitude', 'status', 'num_trips_lag']]
y = station_usage['num_trips']

# 80-20 Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print(f"The training set has {X_train.shape[0]} samples.")
#print(f"The testing set has {X_test.shape[0]} samples.")

# Models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
}

results = {}

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Store the results
    results[name] = {'MAE': mae, 'RMSE': rmse}

# Print out the performance of each model
for model_name, metrics in results.items():
    print(f"{model_name} - MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}")

# Hyperparameters for Random Forest and XGBoost
rf_param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(10, 30),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10)
}

xgb_param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 10),
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
}

# Initialize RandomizedSearchCV
rf_search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=42), 
                               param_distributions=rf_param_dist, 
                               n_iter=10, cv=5, random_state=42, n_jobs=-1)

xgb_search = RandomizedSearchCV(estimator=xgb.XGBRegressor(random_state=42), 
                                param_distributions=xgb_param_dist, 
                                n_iter=10, cv=3, random_state=42, n_jobs=-1)


# Train models
rf_search.fit(X_train, y_train)
xgb_search.fit(X_train, y_train)

# Best models
best_rf = rf_search.best_estimator_
best_xgb = xgb_search.best_estimator_

# Evaluate models
rf_pred = best_rf.predict(X_test)
xgb_pred = best_xgb.predict(X_test)

# MAE and RMSE
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

# Results
print(f'Random Forest (Tuned) - MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}')
print(f'XGBoost (Tuned) - MAE: {xgb_mae:.2f}, RMSE: {xgb_rmse:.2f}')

#print(station_usage['num_trips'].describe())

# Make predictions with the best model (XGBoost Tuned)
y_pred_final = best_xgb.predict(X_test) 

# Compare results
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_final})

comparison['Station ID'] = X_full.loc[X_test.index, 'start_station_id'].values
comparison['Hour'] = X_full.loc[X_test.index, 'hour'].values
comparison['Day of Week'] = X_full.loc[X_test.index, 'day_of_week'].values

print(comparison.head())

# Plot actual vs predicted values
r2 = r2_score(y_test, y_pred_final)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_final, color='blue', alpha=0.5)
plt.plot([0, max(y_test)], [0, max(y_pred_final)], color='red', linestyle='--')  
plt.text(1, 0.9 * max(y_pred_final), f"$R^2$: {r2:.2f}", 
         fontsize=12, color='green')
plt.title('Actual vs Predicted Number of Trips')
plt.xlabel('Actual Trips')
plt.ylabel('Predicted Trips')
plt.show()

# Save the best model
#joblib.dump(best_xgb, 'model.pkl')
