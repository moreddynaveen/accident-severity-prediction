import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(r'D:/Accident_Project/road.csv')

df['Educational_level'].value_counts().plot(kind='bar')
plt.xlabel('Educational_level')
plt.ylabel('Counts')
plt.show()