# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 23:48:31 2021

@author: 54651
"""
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt 
import plotly as py

import warnings
import os
from tsfeatures import tsfeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


warnings.filterwarnings("ignore")
py.offline.init_notebook_mode(connected = True)


def load_data(database):
    cnx = sqlite3.connect(database)
    return pd.read_sql_query("SELECT * FROM pjm_dom_13KV_loadbus_rtlmps_may2020_april2021", cnx)

def process_data(df):
    df = df.copy(deep=True)
    df = df.iloc[:, 1:]
    df = df[df.type == 'LOAD']
    columns = ['datetime_beginning_ept', 'pnode_id',
               'pnode_name', 'total_lmp_rt']
    df = df[columns]
    df['datetime_beginning_ept'] = pd.to_datetime(df['datetime_beginning_ept'])
    ids = df.pnode_id.unique().tolist()
    ids2=  [34885301, 76427737, 1388604896,  1666105931, 2156111266]
    final= [x for x in ids if x not in ids2]
    df = df[df.pnode_id.isin(final)]
    new_columns = ['pnode_id','datetime_beginning_ept','total_lmp_rt']
    df = df[new_columns]
    df.columns = ['unique_id', 'ds', 'y']
    return df

def basic_stats(df):
    df = df.copy(deep=True)
    df = df.iloc[:, 1:]
    print(df.columns)
    df = df[df.type == 'LOAD']
    columns = ['datetime_beginning_ept', 'pnode_id',
               'pnode_name', 'total_lmp_rt']
    df = df[columns]
    df['datetime_beginning_ept'] = pd.to_datetime(df['datetime_beginning_ept'])
    ids = df.pnode_id.unique()
    all_nodes = []
    for i in ids:
        dummy = df[df.pnode_id == i]
        dummy.sort_values("datetime_beginning_ept", inplace=True)
        # dropping ALL duplicate values
        dummy.drop_duplicates(subset="datetime_beginning_ept",
                              keep=False, inplace=True)
        t = dummy['total_lmp_rt'].describe()
        cnt_over_mean = dummy[dummy['total_lmp_rt'] > t[1]].count()[0]
        results = [i, t[1], t[2], t[3], t[7], cnt_over_mean]
        all_nodes.append(results)
    stats = pd.DataFrame(all_nodes, columns=['pnode_id', 'mean', 
                                             'standard_deviation', 'min', 
                                             'max','count_over_mean'])
    return stats
    
def extract_features(df):
    return tsfeatures(df, freq=24) #data.to_csv('features.csv')

def feature_reduction(features):
    #features = features.copy(deep = True)
    dummy = features['unique_id']
    X = features.drop(['unique_id'], axis = 1)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    pca = PCA(n_components = 2)
    fit = pca.fit_transform(X)
    pca_df = pd.DataFrame(data = fit,columns = ['pca_1', 
                                                'pca_2'])
    output = pd.concat([pca_df, dummy], axis = 1)
    return output

def plot(inertia):
    plt.figure(1 , figsize = (15 ,6))
    plt.plot(np.arange(1 , 11) , inertia , 'o')
    plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
    plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
    plt.show()

def plot_pca(final):
    final = final.copy(deep= True)
    plt.scatter(final.iloc[:, 0], final.iloc[:, 1], c= final['cluster_id'], s=50, cmap='viridis')
    #centers = kmeans.cluster_centers_
    #plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
    
def generate_clusters(data):
    data = data.copy(deep=True)
    if 'unique_id' in data.columns:
        X = data.drop(['unique_id'], axis = 1)
    if 'pnode_id' in data.columns:
        X = data.drop(['pnode_id'], axis = 1)
        
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    inertia = [] # Inertia: Sum of distances of samples to their closest cluster center
    for n in range(1 , 11):
        algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                            tol=0.0001,  random_state= 111  , algorithm='elkan') )
        algorithm.fit(X)
        inertia.append(algorithm.inertia_)
    
    plot(inertia)
    algorithm = (KMeans(n_clusters = 3 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
    algorithm.fit(X)
    labels = algorithm.labels_
    #centroids = algorithm.cluster_centers_
    data['cluster_id'] = labels
    return data   

def test_with_basic():
    df = load_data('LMP_db.db')
    stats = basic_stats(df)
    final = generate_clusters(data=stats)
    final.to_csv('output_with_basic_stats.csv')

def test_with_feature_extraction():
    df = load_data('LMP_db.db')
    df = process_data(df)
    features = extract_features(df)
    final = generate_clusters(data=features)
    final.to_csv('output_with_feature_extraction.csv')
    
def test_with_feature_reduction():
    df = load_data('LMP_db.db')
    df = process_data(df)
    features = extract_features(df)
    pca_df = feature_reduction(features)
    final = generate_clusters(data=pca_df)
    plot_pca(final)
    final.to_csv('output_with_feature_reduction.csv')

if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    test_with_basic()
    test_with_feature_extraction()
    test_with_feature_reduction()

   



