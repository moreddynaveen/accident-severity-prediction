import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
import joblib

# Load your dataset
df = pd.read_csv(r'D:/Accident_Project/road.csv')

# Split the data into features and target
X = df.drop('Accident_severity', axis=1)
y = df['Accident_severity']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode categorical features
encoder = OrdinalEncoder()
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.fit_transform(X_test)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_encoded, y_train)

# Save the model
joblib.dump(model, 'rta_model_deploy3.joblib')

# Save the encoder
joblib.dump(encoder, 'ordinal_encoder2.joblib')

print("Model and encoder saved successfully.")
