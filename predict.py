import pandas as pd
from create_features_df import create_key_feature_df
import cPickle as pickle
from my_model import FraudClassificationModel
import urllib, json

'''
predict.py is a script that runs the fraud detection model on a single
data point.  The work flow is as follows:
    1.  Extract the model from the pickle file
    2.  Pull the data point from the web in json format
    3.  Reformat the data point into a pandas datafram with one row
    4.  Extract the model features from the dataframe
    5.  Calculate the predicted probability of fraud
    6.  Print the predicted probability to the command line
'''

if __name__ == '__main__':
    ######################  Step 1 ######################################
    # extract the model from the pickle file
    with open('data/knn.pkl') as f:
        model = pickle.load(f)

    ######################  Step 2 ######################################
    # url where new information is pulled from
    url = 'http://galvanize-case-study-on-fraud.herokuapp.com/data_point'
    # contact url
    response = urllib.urlopen(url)
    # put json data point into a dictionary
    data = json.loads(response.read())

    ######################  Step 3 ######################################
    # format json data point into a one row pandas dataframe
    item = pd.DataFrame.from_dict(data, orient='index').T

    ######################  Step 4 ######################################
    # extract model features from the dataframe
    feature = create_key_feature_df(item.copy())

    ######################  Step 5 ######################################
    # calculate the probability of fraud
    pred = model.predict(feature)[0]

    ######################  Step 6 ######################################
    # output to command line
    print "Predicted Fraud Probability:", pred
