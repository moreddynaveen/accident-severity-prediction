# import labelencoder from sklearn.preprocessing
import pandas as pd
from sklearn.preprocessing import LabelEncoder

n_df = pd.read_csv(r'D:/Accident_Project/road.csv')


# create labelencoder object
lb = LabelEncoder()
y = n_df['Accident_severity']
lb.fit(y)
y_encoded = lb.transform(y)
print("Encoded labels:",lb.classes_)
y_en = pd.Series(y_encoded)