import pandas as pd

# Assuming df is your DataFrame and 'Time' column is in object type representing datetime strings
df=pd.read_csv(r'D:/Accident_Project/road.csv')

# Convert 'Time' column to datetime type
df['Time'] = pd.to_datetime(df['Time'])

# Create a new DataFrame copy and extract 'Hour_of_Day' from 'Time' column
new_df = df.copy()
new_df['Hour_of_Day'] = new_df['Time'].dt.hour

# Drop the original 'Time' column
n_df = new_df.drop('Time', axis=1)

# Display the first few rows of the new DataFrame
print(n_df.head())
