# import the necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd


from sklearn.metrics import confusion_matrix, classification_report, f1_score

n_df = pd.read_csv(r'D:/Accident_Project/road.csv')

features = ['Day_of_week', 'Number_of_vehicles_involved', 'Number_of_casualties', 'Area_accident_occured',
           'Types_of_Junction', 'Age_band_of_driver', 'Sex_of_driver', 'Educational_level',
           'Vehicle_driver_relation', 'Type_of_vehicle', 'Driving_experience', 'Service_year_of_vehicle', 
           'Type_of_collision', 'Sex_of_casualty', 'Age_band_of_casualty', 'Cause_of_accident']



y = n_df['Accident_severity']
fs = SelectKBest(chi2, k=50)
lb = LabelEncoder()
lb.fit(y)
y_encoded = lb.transform(y)

target = ['Accident_severity']

y = lb.fit_transform(n_df[target])
y_en = pd.Series(y)

X = n_df[features]

encoded_df = pd.get_dummies(X, drop_first=True)

X_new = fs.fit_transform(encoded_df, y_en)
cols = fs.get_feature_names_out()

# Assuming X_n and y_n are defined somewhere above
fs_df = pd.DataFrame(X_new, columns=cols)
y_en = pd.Series(y_encoded)


categorical_features_original = ['Day_of_week', 'Area_accident_occured', 'Types_of_Junction', 
                                 'Age_band_of_driver', 'Sex_of_driver', 'Educational_level', 
                                 'Vehicle_driver_relation', 'Type_of_vehicle', 'Driving_experience', 
                                 'Type_of_collision', 'Sex_of_casualty', 'Age_band_of_casualty', 
                                 'Cause_of_accident']

encoded_columns = encoded_df.columns


categorical_encoded_columns = [col for feature in categorical_features_original for col in encoded_columns if feature in col]

categorical_features_indices = [i for i, col in enumerate(cols) if col in categorical_encoded_columns]

smote = SMOTENC(categorical_features=categorical_features_indices, random_state=42, n_jobs=True)
X_n, y_n = smote.fit_resample(fs_df,y_en)
# train and test split and building baseline model to predict target features
X_trn, X_tst, y_trn, y_tst = train_test_split(X_n, y_n, test_size=0.2, random_state=42)

# modelling using random forest baseline
rf = RandomForestClassifier(n_estimators=800, max_depth=20, random_state=42)
rf.fit(X_trn, y_trn)

# predicting on test data
predics = rf.predict(X_tst)

# Training score
train_score = rf.score(X_trn, y_trn)
print(f'Training Score: {train_score}')

# Testing score
test_score = rf.score(X_tst, y_tst)
print(f'Testing Score: {test_score}')

# Confusion matrix
conf_matrix = confusion_matrix(y_tst, predics)
print('Confusion Matrix:')
print(conf_matrix)

# Classification report
class_report = classification_report(y_tst, predics)
print('Classification Report:')
print(class_report)

# F1 score
f1 = f1_score(y_tst, predics, average='weighted')
print(f'F1 Score: {f1}')

