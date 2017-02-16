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

app = Flask(__name__)
PORT = 8080

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
    dat = counts.values
    fig, ax = plt.subplots(1)
    ax.plot(dat)
    ax.set_xlabel('Fraud Prob')
    ax.set_ylabel('Count')
    fig.suptitle('Count of Events by Fraud Probability', fontsize = 16)
    fig.savefig('images/temp_plot.png')
    return fig

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
    fig = plot_data(df)
    app.run(host='0.0.0.0', port=PORT, debug=True)
