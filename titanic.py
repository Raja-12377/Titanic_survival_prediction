import streamlit as st
import pickle
import pandas as pd
import sklearn
# Load the trained model from the pickle file
filename = 'trained_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Create a Streamlit app
st.title("Titanic Survival Prediction")

# Get user input
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=0, step=1)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, step=1)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, step=1)
fare = st.number_input("Fare", min_value=0.0)
st.write("City names : C = Cherbourg, Q = Queenstown, S = Southampton")
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Create a new DataFrame for the new customer
data = {'Pclass': pclass, 'Sex': 0 if sex == "Female" else 1, 'Age': age, 'SibSp': sibsp, 'Parch': parch, 'Fare': fare, 'Embarked': 0 if embarked == "C" else 1 if embarked == "Q" else 2}
cust_df = pd.DataFrame(data, index=[0])

# Make a prediction using the loaded model
prediction = loaded_model.predict(cust_df)[0]
predictionp = loaded_model.predict_proba(cust_df)[:, 1]

# Display the prediction result
if st.button("Predict"):
    if prediction == 0:
        st.warning("Not Survive")
    else:
        st.success("Survive")
    percent = predictionp[0] * 100
    st.info(f"Survival probability for the person: {percent:.2f}%")
