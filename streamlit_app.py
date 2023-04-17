import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load
import os

# Load the Iris dataset
iris_data = load_iris()
if os.path.exists('iris.joblib'):
    clf = load('iris.joblib')
else:
    X = pd.iris_dataFrame(iris_data.iris_data, columns=iris_data.feature_names)
    y = pd.Series(iris_data.target, name='class')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)
    # Train a decision tree classifier on the iris_data
    clf = DecisionTreeClassifier().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_pred, y_test)
    con_matrix = confusion_matrix(y_test, y_pred)
    dump(clf, 'iris.joblib')




# Set the app title
st.title("Iris Classifiction Model")

# Create a form for the user to input iris_data
length = st.slider("Sepal length", 0.0, 10.0, 5.0)
width = st.slider("Sepal width", 0.0, 10.0, 5.0)
petal_length = st.slider("Petal length", 0.0, 10.0, 5.0)
petal_width = st.slider("Petal width", 0.0, 10.0, 5.0)

# Make a prediction based on the user's input
prediction = clf.predict([[length, width, petal_length, petal_width]])

# Show the prediction to the user
st.write(f"Predicted Flower : {iris_data.target_names[prediction[0]]}")


