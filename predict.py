import pickle
import pandas as pd
import numpy as np

# Load the trained linear regression model
with open('linear_regression_PRINT.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the new CSV file containing the feature values for prediction
df_new = pd.read_csv('image_quality.PRINT.csv')  # Replace 'new_data.csv' with your actual CSV file

# Extract the feature values from the new data
X_new = df_new[['blur', 'brightness', 'contrast']]

# Make predictions using the loaded model
y_pred = model.predict(X_new)

# Calculate the predicted sharpness based on the coefficients
coefficients = model.coef_
predicted_sharpness = np.dot(X_new.values, coefficients)

# Create a DataFrame with the predicted sharpness values
df_pred = pd.DataFrame({'sharpness_prediction': predicted_sharpness})

# Concatenate the original data with the predicted sharpness values
df_output = pd.concat([df_new, df_pred], axis=1)

# Save the DataFrame to a CSV file
df_output.to_csv('predicted_sharpness_print002.csv', index=False)
