import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df2=pd.read_csv(r'D:/Accident_Project/road.csv')


print(df2.info())
print(df2.head())
print(df2.isnull().sum())


df2['Road_surface_type'] = df2['Road_surface_type'].astype('category')
df2['Accident_severity'] = df2['Accident_severity'].astype('category')

sns.countplot(x='Road_surface_type', hue='Accident_severity', data=df2)
plt.show()