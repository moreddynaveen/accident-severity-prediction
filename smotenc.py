from imblearn.over_sampling import SMOTENC
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
n_df = pd.read_csv(r'D:/Accident_Project/road.csv')

features = ['Day_of_week', 'Number_of_vehicles_involved', 'Number_of_casualties', 'Area_accident_occured',
           'Types_of_Junction', 'Age_band_of_driver', 'Sex_of_driver', 'Educational_level',
           'Vehicle_driver_relation', 'Type_of_vehicle', 'Driving_experience', 'Service_year_of_vehicle', 
           'Type_of_collision', 'Sex_of_casualty', 'Age_band_of_casualty', 'Cause_of_accident']

target = ['Accident_severity']

lb = LabelEncoder()
y = lb.fit_transform(n_df[target])
y_en = pd.Series(y)

X = n_df[features]
encoded_df = pd.get_dummies(X, drop_first=True)

fs = SelectKBest(chi2, k=50)
X_new = fs.fit_transform(encoded_df, y_en)
cols = fs.get_feature_names_out()

fs_df = pd.DataFrame(X_new, columns=cols)

categorical_features_original = ['Day_of_week', 'Area_accident_occured', 'Types_of_Junction', 
                                 'Age_band_of_driver', 'Sex_of_driver', 'Educational_level', 
                                 'Vehicle_driver_relation', 'Type_of_vehicle', 'Driving_experience', 
                                 'Type_of_collision', 'Sex_of_casualty', 'Age_band_of_casualty', 
                                 'Cause_of_accident']

encoded_columns = encoded_df.columns

categorical_encoded_columns = [col for feature in categorical_features_original for col in encoded_columns if feature in col]

categorical_features_indices = [i for i, col in enumerate(cols) if col in categorical_encoded_columns]




 
smote = SMOTENC(categorical_features=categorical_features_indices, random_state=42, n_jobs=-1)


X_n, y_n = smote.fit_resample(fs_df, y_en)

print(X_n.shape, y_n.shape)

print(y_n.value_counts())
