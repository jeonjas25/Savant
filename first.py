import pybaseball as pb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

# Fetch batting stats for 2022 and 2023
data_2022 = pb.batting_stats(2022)
data_2023 = pb.batting_stats(2023)

# Combine data into one DataFrame
data = pd.concat([data_2022, data_2023])

# Drop rows with missing target values (batting average)
data.dropna(subset=['AVG'], inplace=True)

# Fill missing values in other columns if necessary
data.fillna(0, inplace=True)

# Include player names
names = data['Name']

# Select relevant columns for features and target
features = data[['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'BB', 'SO', 'SB', 'CS']]
target = data['AVG']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(features, target, names, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
predictions = model.predict(X_test)

# Calculate the Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')

# Plotting the actual vs predicted batting averages as a bar chart
x = np.arange(len(names_test))  # The label locations
width = 0.35  # The width of the bars

fig, ax = plt.subplots(figsize=(14, 8))
rects1 = ax.bar(x - width/2, y_test, width, label='Actual AVG')
rects2 = ax.bar(x + width/2, predictions, width, label='Predicted AVG')

# Add some text for labels, title, and custom x-axis tick labels
ax.set_xlabel('Player')
ax.set_ylabel('Batting Average')
ax.set_title('Actual vs Predicted Batting Averages')
ax.set_xticks(x)
ax.set_xticklabels(names_test, rotation=90)
ax.legend()

fig.tight_layout()

plt.show()

# Preparing for 2024 projections
# Assuming 2024 projections are based on the most recent stats (2023)

# Extract features for 2023 data
features_2023 = data_2023[['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'BB', 'SO', 'SB', 'CS']]
names_2023 = data_2023['Name']

# Make projections for 2024 based on 2023 data
predictions_2024 = model.predict(features_2023.fillna(0))  # Fill any NaN values

# Combine the names with their predicted AVG for 2024
projections_2024 = pd.DataFrame({
    'Name': names_2023,
    'Predicted AVG 2024': predictions_2024
})

# Print the projections DataFrame
print(projections_2024)
