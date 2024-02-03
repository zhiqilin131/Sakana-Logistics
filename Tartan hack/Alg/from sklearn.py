from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# Step 1: Generate Synthetic Data
# Setting a random seed for reproducibility
np.random.seed(0)

# Generate synthetic data for 100 observations with 3 sets of risk inputs, each having 5 columns
num_observations = 100
risk_input1 = np.random.rand(num_observations, 5)  # First set of risk inputs
risk_input2 = np.random.rand(num_observations, 5) * 2  # Second set, scaled differently
risk_input3 = np.random.rand(num_observations, 5) * 3  # Third set, scaled differently

# Combine these inputs into a single DataFrame for easier manipulation
X = np.hstack((risk_input1, risk_input2, risk_input3))  # Combine into a single array
y = X.dot(np.random.rand(15)) + np.random.randn(num_observations) * 0.5  # Target variable

# Step 2: Prepare the Data
# Splitting the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build the Regression Model
# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Evaluate the Model
# Predict the target variable for the testing set
y_pred = model.predict(X_test)

# Calculate and print the mean squared error between the predicted and actual values
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
