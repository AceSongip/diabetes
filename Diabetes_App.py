# -*- coding: utf-8 -*-
"""
Created on Fri May 13 20:49:08 2022

@author: aceso
"""

#%%Modules
import pickle
import os
import numpy as np
import streamlit as st

# Constant
SCALER_SAVE_PATH = os.path.join(os.getcwd(), "static", "mm_scaler.pkl")
MODEL_PATH = os.path.join(os.getcwd(), "static", "model.pkl")

#%% Loading
# Scaler
with open(SCALER_SAVE_PATH, 'rb') as f:
    mm_scaler = pickle.load(f)
# Model
with open(MODEL_PATH, 'rb') as g:
    random_forest = pickle.load(g)
    
diabetes_chance = {0:"negative", 1:"positive"}

#%% deployment

# Insert data and scale it
patience_info = np.array([5,116,74,0,0,25.6,0.201,30])
patience_info = mm_scaler.transform(np.expand_dims(patience_info, axis=0))

# predict using model
new_pred = random_forest.predict(patience_info)
if np.argmax(new_pred) == 1:
    new_pred = [0,1]
    print(diabetes_chance[np.argmax(new_pred)])
else:
    new_pred = [1,0]
    print(diabetes_chance[np.argmax(new_pred)])
    
#%% Build app using streamlit <-- refers streamlit form on google
with st.form('Diabetes Prediction Form'):
    st.write("Patient's Info")
    pregnancies = int(st.number_input("Insert Time of Pregnant")) # add int because not float
    glucose = st.number_input("Glucose") # not need int because of float value
    bp = st.number_input("Blood Pressure")
    skin_thick = st.number_input("Skin_Thickness")
    insulin_level = st.number_input("Insulin Level")
    bmi = st.number_input("BMI")
    diabetes = st.number_input("Diabetes")
    age = int(st.number_input("Age"))
    
    submitted = st.form_submit_button('Submit')
    
    if submitted == True:
        patience_info = np.array([pregnancies,glucose,bp,skin_thick,insulin_level,
                                  bmi,diabetes,age])
        patience_info = mm_scaler.transform(np.expand_dims(patience_info, axis=0))
        new_pred = random_forest.predict(patience_info)
        if np.argmax(new_pred) == 1:
            st.warning(f"You are {diabetes_chance[np.argmax(new_pred)]} diabetes")
        else:
            st.snow()
            st.success(f"You are {diabetes_chance[np.argmax(new_pred)]} diabetes")
    

# copy the folder path you store the file
# make sure the env already activated
# go to command prompt the past the folder path,(cd <path>)
# streamlit run diabetes_aoo.py
