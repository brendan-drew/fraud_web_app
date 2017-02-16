import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from pymongo import MongoClient
import pandas as pd
from flask import Flask, request, render_template
import json
import requests
import socket
import time
from datetime import datetime
import os

'''
This file runs a flask app with the "Hello, World!" app and the fraud detection dashboard.
Includes:
- Function to get new data from the fraud_detections Mongo database
- Function to add plots to our dashboard
- Flask app for posting data to web
'''

app = Flask(__name__)
PORT = 8105

def get_new_data():
    # time_threshold =
    # Connect to the fraud detection database
    client = MongoClient()
    db = client['fraud_detection']
    collection = db['fraud_collection']
    # Create a MongoDB query
    dat = collection.find()
    lst = []
    for x in dat:
        lst.append([x['timestamp'], x['fraud_prob']])
    df = pd.DataFrame(lst, columns = ['timestamp', 'fraud_prob'])
    return df

def plot_data(df):
    counts = df.groupby('fraud_prob').count()
    labels = counts.index
    dat = counts.values
    fig, ax = plt.subplots(1)
    ax.bar(left = labels, height = dat)
    ax.set_xlabel('Probability of Fraud')
    ax.set_ylabel('Number of Events')
    fig.suptitle('Count of Events by Fraud Probability', fontsize = 16)
    fig.savefig('static/temp_plot.png')
    fig.show()
    return dat

@app.route('/')
def index():
    return '''
    <h1>Welcome to the fraud detection website</h1>
    <form action = '/score'>
        <input type = 'submit' value = 'Get model scores'>
    </form>
    '''

@app.route('/score')
def score():
    return '''
    <body>
        <img src = '/static/temp_plot.png' alt = 'our_plot'>
        <p>This is our plot</p>
    </body>
    '''


@app.route('/hello')
def say_hello():
    return '''
    <h1>Hello, World!</h1>
    '''

if __name__ == '__main__':
    # Query data from MongoDB
    df = get_new_data()
    dat = plot_data(df)
    app.run(host='ec2-54-164-186-225.compute-1.amazonaws.com', port=PORT, threaded = True)
