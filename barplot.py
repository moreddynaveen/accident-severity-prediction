# importing pandas
import pandas as pd
# importing matplotlib
import matplotlib.pyplot as plt

# using pandas read_csv function to load the dataset
data = pd.read_csv(r'D:/Accident_Project/road.csv')
df=pd.DataFrame(data)
# Print the counts of each category in the 'Accident_severity' column
print(df['Accident_severity'].value_counts())
# Plot a bar plot of the counts
df['Accident_severity'].value_counts().plot(kind='bar')
# Show the plot
plt.xlabel('Accident Severity')
plt.ylabel('Counts')
plt.title('Counts of Accident Severity')
plt.show()

