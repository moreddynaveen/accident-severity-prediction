import pandas as pd
import matplotlib.pyplot as plt

# Example DataFrame
data = pd.read_csv(r'D:/Accident_Project/road.csv')
df = pd.DataFrame(data)

# Print the counts of each category in the 'Accident_severity' column
print(df['Accident_severity'].value_counts())

# Plot a pie chart of the counts
df['Accident_severity'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, counterclock=False)

# Show the plot
plt.ylabel('')  # Hide y-label for better appearance
plt.title('Distribution of Accident Severity')
plt.show()