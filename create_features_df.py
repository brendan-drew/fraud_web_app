import pandas as pd
import numpy as np
import random
import cPickle as pickle
from pdb import set_trace
from my_model import FraudClassificationModel
import sys
import ast
import urllib, json

### take dataframe and replace columns with key features
def create_key_feature_df(df, filedir = 'data/'):
    '''
    INPUT:
        df - pandas dataframe containing the raw fraud prediction data
        filedir - string containing the directory of the pickled lists
                  of high risk countries and email domains.
    OUTPUT:
        pandas dataframe with with the engineered features used to build
        models or run models on new data points
    '''
    orig_columns = df.columns

    #*** add new columns
    # - Has logo (Y/N)
    df['kf_logo'] = df['has_logo']
    # - Organization description listed (Y/N)
    df['kf_org_desc'] = [int(len(x)>0) for x in df['org_desc']]
    # - High-risk country (based on % fraud, e.g. 30%)
    cntry_thresh = .30
    df['country'].replace(to_replace=[None, ''],value='None', inplace=True)
    with open(filedir + 'country_fraud.pkl') as f:
        country_fraud = pickle.load(f)
    kf_hr_country = []
    for x in df['country'].astype(str):
        try:
            kf_hr_country.append(int(country_fraud[x] > cntry_thresh))
        except:
            kf_hr_country.append(0)
    df['kf_hr_country'] = kf_hr_country
    # - Body length < 10000
    body_len = 10000
    df['kf_body_len'] = (df['body_length'] < body_len).astype(int)
    # - High risk email domain (based on % fraud - e.g. 30%)
    email_thresh = .30
    with open(filedir + 'email_fraud.pkl') as f:
        email_fraud = pickle.load(f)
    kf_email_dom = []
    for x in df['email_domain'].astype(str):
        try:
            kf_email_dom.append(int(email_fraud[x] > email_thresh))
        except:
            kf_email_dom.append(0)
    df['kf_email_dom'] = kf_email_dom
    # - User age < 1000
    user_thresh = 1000
    df['kf_user_age'] = (df['user_age'] <= user_thresh).astype(int)
    # - Num payouts < 30
    payout_thresh = 30
    df['kf_payouts'] = (df['num_payouts'] <= payout_thresh).astype(int)
    # - US = 0 else = 1
    df['kf_intl'] = (df['country'] != 'US').astype(int)
    # - event local to user
    df['kf_local'] = (df['country'] == df['venue_country']).astype(int)
    # - name length
    name_thresh = 20
    df['kf_name_len'] = (df['name_length'] <= name_thresh).astype(int)
    # - user type
    df['kf_user_type'] = (df['user_type'] == 1).astype(int)
    # - name is all upper chase
    df['kf_upper_name'] = (df['name'].str.isupper()).astype(int)
    # - org_twitter is twitter
    df['kf_twitter_org'] = (df['org_twitter'] != 0.0).astype(int)
    # - channels if 0
    df['kf_channels'] = (df['channels'] == 0).astype(int)
    # - if currency is EUR or GBP
    df['kf_euro_gb_currency'] = ((df['currency']=='EUR') | (df['currency']=='GBP')).astype(int)

    #*** delete original columns other than 'fraud'
    for col in orig_columns:
        if col != 'fraud':
            df.drop(col, axis=1, inplace=True)

    idx = np.arange(1, df.shape[0]+1)
    df['index'] = idx
    df.set_index('index', drop=True, inplace=True)

    return df

# ### copy in data.json file and prepare data for analysis
def prepare_df(df, filedir = 'data/', train=True):

    #*** if preparing training data - add 'fraud' column and created fraud dependent tables
    if train:
        #*** add fraud column
        fraud  = ['fraudster' in x for x in df['acct_type']]
        df['fraud'] = fraud
        df['country'].replace(to_replace=[None, ''],value='None', inplace=True)
        country_fraud = df.groupby('country')['fraud'].mean()
        with open(filedir + 'country_fraud.pkl', 'w') as f:
            pickle.dump(country_fraud, f)
        country_fraud.to_csv(filedir + 'country_fraud.csv')
        email_fraud = df.groupby('email_domain')['fraud'].mean()
        with open(filedir + 'email_fraud.pkl', 'w') as f:
            pickle.dump(email_fraud, f)

    #*** convert epoch dates to datetime columns
    date_columns = ['event_created', 'event_start', 'event_end', 'approx_payout_date', 'user_created', 'event_published']

    for col in date_columns:
        df[col] = pd.to_datetime(df[col], unit='s')

    return df

def create_raw_df_sample(df, path_name, samples = 3):
    sample = df.iloc[random.sample(df.index, 3)]
    sample.to_json(path_name)

#*** Create data for model and sampling
def create_data(file):
    #*** read json file
    df = pd.read_json(file)

    #*** create database sample
    create_raw_df_sample(df, path_name = filedir + 'test_script_examples.json', samples = 3)

    #*** prepare df for training of model
    df = prepare_df(df)
    #    convert training df columns to features
    df = create_key_feature_df(df)
    #    store model training df to csv file
    df.to_csv(filedir + 'key_feature_df.csv')


if __name__ == '__main__':
    filedir = 'data/'

    #*** create initial DF for fitting and test sample
    # filename = 'data.json'
    # file = filedir + filename
    # create_data(file)

    url = 'http://galvanize-case-study-on-fraud.herokuapp.com/data_point'
    response = urllib.urlopen(url)
    data = json.loads(response.read())
    rdf = pd.DataFrame.from_dict(data, orient='index').T

    #*** read raw test df
    # filename = 'test_script_examples.json'
    # # #filename = 'data_point.json'
    # rdf = pd.read_json(filedir + filename)

    fin_rdf = rdf.copy()

    #*** prepare df of test_data
    fin_rdf = prepare_df(fin_rdf, train=False)
    #    convert test df columns to features
    fin_rdf = create_key_feature_df(fin_rdf)
    #    store test df to pickel file
    with open('data/test_feature_df.pkl', 'w') as f:
        pickle.dump(fin_rdf, f)

    #*** get test df from pickel file
    with open(filedir + 'test_feature_df.pkl') as f:
        fin_rdf = pickle.load(f)
    with open('model.pkl') as f:
        model = pickle.load(f)
    for i in range(len(fin_rdf.index)):
        sys.stdout.write(str(rdf.iloc[i]) + '\n')
        sys.stdout.write(str(model.predict(fin_rdf.iloc[i].reshape(1,-1))) + '\n')
