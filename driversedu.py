import pandas as pd
import matplotlib.pyplot as plt

# Example DataFrame
data = pd.read_csv(r'D:/Accident_Project/road.csv')
df=pd.DataFrame(data)

# Print the counts of each category in the 'Educational_level' column
print(df['Educational_level'].value_counts())

# Plot a bar plot of the counts
df['Educational_level'].value_counts().plot(kind='bar')

# Add labels and title
plt.xlabel('Educational Level')
plt.ylabel('Counts')
plt.title('Counts of Educational Levels of Car Drivers')

# Show the plot
plt.show()