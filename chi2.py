from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
n_df = pd.read_csv(r'D:/Accident_Project/road.csv')

features = ['Day_of_week','Number_of_vehicles_involved','Number_of_casualties','Area_accident_occured',
           'Types_of_Junction','Age_band_of_driver','Sex_of_driver','Educational_level',
           'Vehicle_driver_relation','Type_of_vehicle','Driving_experience','Service_year_of_vehicle','Type_of_collision',
           'Sex_of_casualty','Age_band_of_casualty','Cause_of_accident']

X = n_df[features]
y = n_df['Accident_severity']
lb = LabelEncoder()

y_encoded = lb.fit_transform(y)
y_en = pd.Series(y_encoded)

encoded_df = pd.get_dummies(X, drop_first=True)


fs = SelectKBest(chi2, k=50)


X_new = fs.fit_transform(encoded_df, y_en)

# Take the selected features
cols = fs.get_feature_names_out()

# convert selected features into dataframe
fs_df = pd.DataFrame(X_new, columns=cols)

print(fs_df)
