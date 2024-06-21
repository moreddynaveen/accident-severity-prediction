import joblib
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
n_df = pd.read_csv(r'D:/Accident_Project/road.csv')

# Select features
features = ['Day_of_week', 'Number_of_vehicles_involved', 'Number_of_casualties', 'Area_accident_occured',
            'Types_of_Junction', 'Age_band_of_driver', 'Sex_of_driver', 'Educational_level',
            'Vehicle_driver_relation', 'Type_of_vehicle', 'Driving_experience', 'Service_year_of_vehicle', 
            'Type_of_collision', 'Sex_of_casualty', 'Age_band_of_casualty', 'Cause_of_accident']

featureset_df = n_df[features]
feature_df = featureset_df.copy()

# Encode categorical features
new_fea_df = feature_df[['Type_of_collision', 'Age_band_of_driver', 'Sex_of_driver',
                         'Educational_level', 'Service_year_of_vehicle', 'Day_of_week',
                         'Area_accident_occured']]

oencoder2 = OrdinalEncoder()
encoded_df3 = pd.DataFrame(oencoder2.fit_transform(new_fea_df))
encoded_df3.columns = new_fea_df.columns

# Save the ordinal encoder object for inference pipeline
joblib.dump(oencoder2, "ordinal_encoder2.joblib")

# Combine encoded categorical features with numerical features
s_final_df = pd.concat([feature_df[['Number_of_vehicles_involved', 'Number_of_casualties']], 
                        encoded_df3], axis=1)

# Encode target variable
target = ['Accident_severity']
lb = LabelEncoder()
y = lb.fit_transform(n_df[target].values.ravel())
y_en = pd.Series(y)

# Split the data into training and testing sets
X_trn2, X_tst2, y_trn2, y_tst2 = train_test_split(s_final_df, y_en, test_size=0.2, random_state=42)

# Train the Random Forest model
rf = RandomForestClassifier(n_estimators=700, max_depth=20, random_state=42)
rf.fit(X_trn2, y_trn2)

# Save the trained model
train=joblib.dump(rf, "rta_model_deploy3.joblib", compress=9)
print(train)
