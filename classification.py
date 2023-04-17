import streamlit as st
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Wine Recognition dataset
wine = load_wine()

# Create a DataFrame from the dataset
df = pd.DataFrame(wine.data, columns=wine.feature_names)
# df.describe().to_csv('wine.csv')
df['target'] = wine.target

# Set the page title
st.set_page_config(page_title='Wine Recognition Classification')

# Set the page header
st.title('Wine Recognition Classification')

# Display the raw data
if st.checkbox('Show raw data'):
    st.write(df)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[wine.feature_names], df['target'], random_state=0)

# Create and train the model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Display the accuracy score
st.write('Accuracy:', accuracy)

# Display feature importance
st.header('Feature Importance')
feature_importance = pd.DataFrame({
    'feature': wine.feature_names,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)
st.bar_chart(feature_importance.set_index('feature'))

# Display confusion matrix
st.header('Confusion Matrix')
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
st.write(confusion_matrix)

st.header('Let\'s Predict')

alcohol_concentration = st.slider("Alcohol Concentration", 11.000, 13.00, 15.00)
malic_acid = st.slider("Malic Acid", 0.50, 1.86, 6.00)
ash = st.slider("Ash", 1.00, 2.35, 3.50)
acl = st.slider("Alcalinity", 10.0, 19.5, 30.0)

phenol = st.slider("Phenol", 70,98, 162 )
flavanoids = st.slider("Flavanoids", 0.30, 2.13, 5.50)
nonflavanoid_phenols = st.slider("Nonflavanoid_phenols", 0.10, 0.34, 0.70)
proanth = st.slider("Proanth", 0.40, 1.55, 3.60)

color_int = st.slider("Color Intensity", 1.20, 4.70, 13.00)
hue = st.slider("Hue", 0.40, 0.96, 1.80)
dilution = st.slider("Dilution", 1.20, 2.78, 4.00)
proline = st.slider("proline", 278.0, 673.5, 1680.0)

prediction = clf.predict([[alcohol_concentration,malic_acid,ash,acl,phenol,flavanoids,nonflavanoid_phenols,
                           proanth,color_int,hue,dilution,proline]])

# Show the prediction to the user
st.write(f"Predicted Wine Class: {wine.target[prediction[0]]}")

