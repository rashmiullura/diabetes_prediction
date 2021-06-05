#This program detects if someone has diabetes using machine learning

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

st.write("""
# Diabetes Detection
Detect if someone has diabetes using machine learning and python!
""")

#open and display an image
image = Image.open('C:\\Users\RASHMI\Documents\Projects\image1.jpeg')
st.image(image, caption='ML', use_column_width=True)

df = pd.read_csv('C:\\Users\RASHMI\Documents\StreamLit\diabetes\diabetes.csv')

#Set a subheader
st.subheader('Data Information')

#Show the data as a table
st.dataframe(df)

#Show statistics on the data
st.write(df.describe())

#Show the data as a chart
chart = st.bar_chart(df)

#Split the dataset into independent x and dependent y variables
x = df.iloc[:, 0:8].values
y = df.iloc[:,-1].values

#split the dataset into 75% training and 25% testing
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)

#get the feature input from the user
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    #above value starts at 0 and ends at 17 and default at 3
    glucose = st.sidebar.slider('glucose',0,195,117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122,72)
    skin_thickness = st.sidebar.slider('skin_thickness',0,99,23)
    insulin = st.sidebar.slider('insulin',0.0, 846.0,30.0)
    BMI = st.sidebar.slider('BMI',0.0,67.1,32.0)
    DPF = st.sidebar.slider('DPF',0.078,2.42,0.3725)
    age = st.sidebar.slider('age', 21,81,29)

    #store a dictionary into a variable
    user_data ={
    	'pregnancies':pregnancies,
    	'glucose':glucose,
    	'blood_pressure': blood_pressure,
    	'skin_thickness': skin_thickness,
    	'insulin':insulin,
    	'BMI':BMI,
    	'DPF':DPF,
    	'age':age
    }

    #Transform the data into data frame 
    features = pd.DataFrame(user_data, index = [0])
    return features

#Store the user input into a variable
user_input = get_user_input()

#Set a subheader and display the user input
st.subheader('User Input:')
st.write(user_input)

#Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(x_train,y_train)

#show the models metrics
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(y_test, RandomForestClassifier.predict(x_test))*100)+'%')

#store the models preidiction in a variable
prediction = RandomForestClassifier.predict(user_input)



#streamlit run "C:\Users\RASHMI\Documents\Projects\webpage1.py"