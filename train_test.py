from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import pickle

# Load the CSV file into a DataFrame
df = pd.read_csv('image_quality.PRINT.csv')

# The target values are the sharpness scores
y = df['sharpness']
# The features are the other image metrics
X = df[['blur', 'brightness', 'contrast']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = model.predict(X_test)

# Print the coefficients
print('Coefficients: \n', model.coef_)
# Print the mean squared error
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
# Print the Root mean squared error
print('Root Mean squared error: %.2f' % np.sqrt(mean_squared_error(y_test, y_pred)))
# Print the Mean absolute error
print('Mean absolute error: %.2f' % mean_absolute_error(y_test, y_pred))
# Print the coefficient of determination: 1 is perfect prediction
print('Coefficient of determination (R^2): %.2f' % r2_score(y_test, y_pred))

# Save the trained model to a file
filename = 'linear_regression_PRINT.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)

# Load the trained model from the file
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Use the loaded model to make predictions
new_X = [[100, 200, 300]]  # Example new feature values
new_y_pred = loaded_model.predict(new_X)
print('Predicted sharpness for new data:', new_y_pred)
