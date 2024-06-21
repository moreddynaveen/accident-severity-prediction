# importing pandas
import pandas as pd

# using pandas read_csv function to load the dataset
df = pd.read_csv(r'D:/Accident_Project/road.csv')
miss=df.isnull().sum()
print(miss)