import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def predict_cardio():
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

    # Prompt the user for input values
    age = int(input("Enter the age: (By Days)"))
    gender = int(input("Enter the gender (1 - women, 2 - men): "))
    height = int(input("Enter the height (in cm): "))
    weight = int(input("Enter the weight (in kg): "))
    ap_hi = int(input("Enter the systolic blood pressure: "))
    ap_lo = int(input("Enter the diastolic blood pressure: "))
    cholesterol = int(input("Enter the cholesterol level (1: normal, 2: above normal, 3: well above normal): "))
    gluc = int(input("Enter the glucose level (1: normal, 2: above normal, 3: well above normal): "))
    smoke = int(input("Enter the smoking status (0 for non-smoker, 1 for smoker): "))
    alco = int(input("Enter the alcohol intake status (0 for non-drinker, 1 for drinker): "))
    active = int(input("Enter the activity level (0 for inactive, 1 for active): "))

    # Create a DataFrame with the user input values
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'height': [height],
        'weight': [weight],
        'ap_hi': [ap_hi],
        'ap_lo': [ap_lo],
        'cholesterol': [cholesterol],
        'gluc': [gluc],
        'smoke': [smoke],
        'alco': [alco],
        'active': [active]
    })

    # Make a prediction on the input data
    cardio_pred = model.predict(input_data)
    if cardio_pred[0] == 0:
        prediction = "No"
    else:
        prediction = "Yes"

    print("Prediction: Cardio =", prediction)

predict_cardio()
