import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("diabetes_prediction_dataset.csv")


# Female -1 Male -2 Other - 3
# No Info - 0 Never - 1 Current - 2 Former - 3 Ever - 4 Not 2 - 5


# Separate the features (X) and the target variable (y)
X = data.drop("diabetes", axis=1)
y = data["diabetes"]

# Create an instance of the Logistic Regression model
model = LogisticRegression(max_iter=2000)

# Train the model on the entire dataset
model.fit(X, y)

# Make predictions on the entire dataset
y_pred = model.predict(X)

# Print the actual and predicted values for each sample
for i in range(50):
    print("Actual:", y.iloc[i], "Predicted:", y_pred[i])

# Evaluate the model's accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)

# Create a DataFrame to hold the actual and predicted values
results = pd.DataFrame({'Actual': y, 'Predicted': y_pred})

# Create a scatter plot
plt.scatter(results['Predicted'], results['Actual'])

# Set plot labels and title
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')
plt.title('Actual vs. Predicted Values')

# Show the plot
plt.show()
