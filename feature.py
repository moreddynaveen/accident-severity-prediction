import pandas as pd

# Load the data into a DataFrame
n_df = pd.read_csv(r'D:/Accident_Project/road.csv')

# Print the columns to verify the names
print(n_df.columns)

# Corrected feature list based on the actual DataFrame columns
features = ['Day_of_week', 'Number_of_vehicles_involved', 'Number_of_casualties', 'Area_accident_occured',
            'Types_of_Junction', 'Age_band_of_driver', 'Sex_of_driver', 'Educational_level',
            'Vehicle_driver_relation', 'Type_of_vehicle', 'Driving_experience', 'Service_year_of_vehicle', 'Type_of_collision',
            'Sex_of_casualty', 'Age_band_of_casualty', 'Cause_of_accident']

# Check if 'Hour_of_Day' is in the DataFrame
if 'Hour_of_Day' in n_df.columns:
    features.append('Hour_of_Day')

# New dataframe generated
featureset_df = n_df[features]
target = n_df['Accident_severity']

# Create a copy of dataframe featureset_df to handle the missing values
feature_df = featureset_df.copy()

# NaN are missing because service info might not be available, we will fill as 'Unknowns'
feature_df['Service_year_of_vehicle'] = feature_df['Service_year_of_vehicle'].fillna('Unknown')
feature_df['Types_of_Junction'] = feature_df['Types_of_Junction'].fillna('Unknown')
feature_df['Area_accident_occured'] = feature_df['Area_accident_occured'].fillna('Unknown')
feature_df['Driving_experience'] = feature_df['Driving_experience'].fillna('unknown')
feature_df['Type_of_vehicle'] = feature_df['Type_of_vehicle'].fillna('Other')
feature_df['Vehicle_driver_relation'] = feature_df['Vehicle_driver_relation'].fillna('Unknown')
feature_df['Educational_level'] = feature_df['Educational_level'].fillna('Unknown')
feature_df['Type_of_collision'] = feature_df['Type_of_collision'].fillna('Unknown')


# Features information
feature_df.info()
