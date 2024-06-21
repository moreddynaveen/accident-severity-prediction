import pandas as pd

# Load the data into a DataFrame
n_df = pd.read_csv(r'D:/Accident_Project/road.csv')

# List of categorical features to encode using one-hot encoding 
features = ['Day_of_week','Number_of_vehicles_involved','Number_of_casualties','Area_accident_occured',
           'Types_of_Junction','Age_band_of_driver','Sex_of_driver','Educational_level',
           'Vehicle_driver_relation','Type_of_vehicle','Driving_experience','Service_year_of_vehicle','Type_of_collision',
           'Sex_of_casualty','Age_band_of_casualty','Cause_of_accident']


# Select the input features (X) and target (y)
X = n_df[features]  # Assuming 'feature_df' and 'n_df' are the same or 'feature_df' is derived from 'n_df'
y = n_df['Accident_severity']

# Perform one-hot encoding on the input features
encoded_df = pd.get_dummies(X, drop_first=True)

# Print the shape of the encoded DataFrame
print(encoded_df.shape)
