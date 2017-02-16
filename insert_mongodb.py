from pymongo import MongoClient
import pandas as pd
from create_features_df import create_key_feature_df
import cPickle as pickle
from my_model import FraudClassificationModel
import urllib, json
from datetime import datetime
import time

def insert_item(item, pred, collection):
    '''
    INPUT:
        item - single row of a pandas dataframe containing the raw input data
        pred - float containing the predicted probability of fraud for the raw data
        collection - MongoDB collection that data will be inserted into
    '''

    # create an empty dictionary that will be entered into the database
    d = {}
    # loop through each column in item and append the entry to the dictionary
    for k in xrange(item.shape[1]):
        d[item.columns[k]] = item.iloc[0, k]
    # add the fraud probability to the dictionary
    d['fraud_prob'] = pred
    # add timestampe
    d['timestamp'] = datetime.now()
    # insert the dictionary into the database
    collection.insert_one(d)

if __name__ == '__main__':
    # open Mongo Client and create variables for database and collection
    client = MongoClient()
    db = client['fraud_detection']
    collection = db['fraud_collection']

    # extract the model from the pickle file
    with open('data/knn.pkl') as f:
        model = pickle.load(f)

    # url where new information is updated
    url = 'http://galvanize-case-study-on-fraud.herokuapp.com/data_point'

    # loop to update database on timer
    time_delta = 5 # seconds
    while True:
        # contact url
        response = urllib.urlopen(url)
        # put json data point into a dictionary
        data = json.loads(response.read())
        # format json data point into a one row pandas dataframe
        item = pd.DataFrame.from_dict(data, orient='index').T
        # extract model features from the dataframe
        feature = create_key_feature_df(item.copy())
        # calculate the probability of fraud
        pred = model.predict(feature)[0]
        # update the MongoDB
        insert_item(item, pred, collection)
        # pause before next iteration of the loop
        time.sleep(time_delta)
        print k
