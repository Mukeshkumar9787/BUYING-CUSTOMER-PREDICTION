# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 22:20:15 2022

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 16:12:13 2022

@author: Admin
"""

from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)
pickle_in=open("dtc_classifier.pkl","rb")
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "WELCOME TO MUKESHKUMAR FLASK SERVER"

@app.route('/predict')
def customer_purchase_prediction():
    
    """ Let's find the customer will purchase or not
    This  is using docstrings for specifications.
    ---
    parameters:
      - name: age
        in: query
        type: number
        required: true
      - name: salary
        in: query
        type: number
        required: true
    responses:
        200:
            description: The Output values
    """
    age=request.args.get("age")
    salary=request.args.get("salary")
    prediction=classifier.predict([[age,salary]])
    return "THE IS THE PREDICTION VALUE IS " + str(prediction)

@app.route('/predict_file',methods=["POST"])
def predict_purchase_values():
    
    """ Let's find the customer will purchase or not
    This  is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    responses:
        200:
            description: The Output values
    """
    
    df_test=pd.read_csv(request.files.get("file"))
    prediction=classifier.predict(df_test)
    return "THE IS THE PREDICTION VALUE for the csv " + str(list(prediction))
    
if __name__=="__main__":
    app.run()
    
