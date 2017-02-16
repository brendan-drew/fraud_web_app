import pandas as pd
import numpy as np
import cPickle as pickle
from sklearn.neighbors import KNeighborsClassifier
import os

'''
This file pickles a classification model for upload to the web
- Defines the FraudClassificationModel class to be pickled
- Defines the functions create_key_features and prepare_df to create the features matrix and target array used to train the classification model
'''

class FraudClassificationModel(object):
    '''
    Classification model class to be uploaded to the web for fraud detection
    '''

    def __init__(self, model):
        '''
        INPUT: Model instantiated with optimal parameters
        OUTPUT: None
        '''
        self.model = model

    def fit(self, X, y):
        '''
        INPUT:
            X: features matrix (np.ndarray)
            y: target array (np.ndarray)
            Note X and y are the output of function create_key_features defined in this script
        OUTPUT: None - fits self.model using the features matrix and target array
        '''
        self.model.fit(X, y)

    def predict(self, x):
        '''
        INPUT: x features of new event to classify (np.ndarray)
        OUTPUT: Probability of fraud (np.ndarray)
        '''
        return self.model.predict_proba(x)[:,1]

def create_key_features(df):
    '''
    INPUT: Pandas dataframe. Assumes dataframe is the output of the prepare_df function below.
    OUTPUT:
        X: Features matrix for classification model (np.ndarray)
        y: Target array for classification model (np.ndarray)
    '''
    # Keep track of the original data columns
    orig_columns = df.columns

    # ADD FEATURES TO DATAFRAME
    # Flag whether the event has a logo
    df['kf_logo'] = df['has_logo']
    # Flag whether organization description listed
    df['kf_org_desc'] = [int(len(x)>0) for x in df['org_desc']]
    # Flag high-risk countries
    cntry_thresh = .30
    df['country'].replace(to_replace=[None, ''],value='None', inplace=True)
    country_fraud = df.groupby('country')['fraud'].mean()
    df['kf_hr_country'] = [int(country_fraud[x] > cntry_thresh) for x in df['country'].astype(str)]
    # Flag body length < 10000
    body_len = 10000
    df['kf_body_len'] = (df['body_length'] < body_len).astype(int)
    # Flag high risk email domain (>30% fraud)
    email_thresh = .30
    email_fraud = df.groupby('email_domain')['fraud'].mean()
    df['kf_email_dom'] = [int(email_fraud[x] > email_thresh) for x in df['email_domain'].astype(str)]
    # Flag user age < 1000
    user_thresh = 1000
    df['kf_user_age'] = (df['user_age'] <= user_thresh).astype(int)
    # Flag num payouts < 30
    payout_thresh = 30
    df['kf_payouts'] = (df['num_payouts'] <= payout_thresh).astype(int)
    # Flag international
    df['kf_intl'] = (df['country'] != 'US').astype(int)
    # Flag event local to user
    df['kf_local'] = (df['country'] == df['venue_country']).astype(int)
    # Flag name length less than 20
    name_thresh = 20
    df['kf_name_len'] = (df['name_length'] <= name_thresh).astype(int)
    # Flag user type 1
    df['kf_user_type'] = (df['user_type'] == 1).astype(int)
    # Flag name all upper class
    df['kf_upper_name'] = (df['name'].str.isupper()).astype(int)
    # Flag org_twitter is twitter
    df['kf_twitter_org'] = (df['org_twitter'] != 0.0).astype(int)
    # Flag channels = 0
    df['kf_channels'] = (df['channels'] == 0).astype(int)
    # Flag currency is EUR or GBP
    df['kf_euro_gb_currency'] = ((df['currency']=='EUR') | (df['currency']=='GBP')).astype(int)

    # DELETE ORIGINAL COLUMNS EXCLUDING FRAUD
    for col in orig_columns:
        if col != 'fraud':
            df.drop(col, axis=1, inplace=True)
    idx = np.arange(1, df.shape[0]+1)
    df['index'] = idx
    df.set_index('index', drop=True, inplace=True)

    # Output X features matrix and y target array
    y = df.pop('fraud').values
    X = df.values
    return X, y


def prepare_df(file, filedir = 'data/'):
    '''
    INPUT: File path
    OUTPUT: Pandas dataframe - passed to create_key_feature_df for feature engineering
    '''
    # Read json file
    df = pd.read_json(file)
    # Add column to flag fraud
    fraud  = ['fraudster' in x for x in df['acct_type']]
    df['fraud'] = fraud
    # Create dataframe of fraud percentage by country
    df['country'].replace(to_replace=[None, ''],value='None', inplace=True)
    country_fraud = df.groupby('country')['fraud'].mean()
    # pickle country fraud dataframe
    with open(filedir + 'country_fraud.pkl', 'w') as f:
        pickle.dump(country_fraud, f)
    # Create dataframe of fraud percentage by email domain
    email_fraud = df.groupby('email_domain')['fraud'].mean()
    # pickle email fraund dataframe
    with open(filedir + 'email_fraud.pkl', 'w') as f:
        pickle.dump(email_fraud, f)
    # Convert epoch dates to datetime columns
    date_columns = ['event_created', 'event_start', 'event_end', 'approx_payout_date', 'user_created', 'event_published']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], unit='s')
    return df

if __name__ == '__main__':
    # Read the json training data to pandas dataframe
    file = os.path.join('data', 'data.json')
    raw_df = prepare_df(file)
    # Convert the raw training data to a features matrix and target array
    X, y = create_key_features(raw_df)
    # Instantiate classification model
    knn = KNeighborsClassifier()
    # Create model class for pickling
    output_model = FraudClassificationModel(knn)
    # Fit the model using available training data
    output_model.fit(X, y)
    # Pickle the model
    with open('model.pkl', 'w') as f:
        pickle.dump(output_model, f)
