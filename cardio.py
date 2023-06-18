import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("cardio_train.csv")

# Separate the features (X) and the target variable (y)
X = data.drop("cardio", axis=1)
y = data["cardio"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42)

# Create an instance of the Logistic Regression model
model = LogisticRegression(max_iter=2000)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Print the actual and predicted values for each test sample
for i in range(20):
    print("Actual:", y_test.iloc[i], "Predicted:", y_pred[i])

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create a DataFrame to hold the actual and predicted values
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Create a scatter plot
plt.scatter(results['Predicted'], results['Actual'])

# Set plot labels and title
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')
plt.title('Actual vs. Predicted Values')

# Show the plot
plt.show()
