# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 03:25:15 2021

@author: 54651
"""


import numpy as np
import pandas as pd
import sqlite3

def load_data(database):
    cnx = sqlite3.connect(database)
    return pd.read_sql_query(
            '''SELECT 
               datetime_beginning_ept, pnode_id,pnode_name, 'total_lmp_rt' 
               FROM 
               pjm_dom_13KV_loadbus_rtlmps_may2020_april2021''', 
           cnx)

def nodal(pnode, df):
    dummy= pd.DataFrame()
    dummy = df[df.pnode_id == pnode]

    dummy.index = dummy.datetime_beginning_ept
    dummy.sort_values("datetime_beginning_ept", inplace=True)
    dummy.drop_duplicates(subset="datetime_beginning_ept",
              keep=False, inplace=True)
    dummy = dummy.resample('1H', base=8).mean()
    dummy = dummy.bfill()
    dummy = pd.DataFrame(dummy, columns=['total_lmp_rt'])
    return dummy['total_lmp_rt'].values

def main():
    df = load_data('LMP_db.db')
    df['datetime_beginning_ept'] = pd.to_datetime(df['datetime_beginning_ept']) 
    
    data = pd.read_csv('output_with_feature_reduction.csv', 
                       usecols=['unique_id','cluster_id'])
    
    #no_of_groups = len(data['cluster_id'].unique())
    total = data.groupby(['cluster_id']).count().iloc[:,0].values.tolist()
    '''
    for c in range(0,no_of_groups):
        cols = ['value']
        exec('df_group_{} = pd.DataFrame(columns = cols)'.format(c)) 
    '''
    df_collection = {}
    
    for i in range(0,data.shape[0]):
        cluster = data.iloc[i,1]
        if cluster == 0:
            df_collection[0][str(pnode)] = nodal(pnode=i, df = df)
        if cluster == 1:
            df_collection[1][str(pnode)] = nodal(pnode=i, df = df)
        if cluster == 2:
            df_collection[2][str(pnode)] = nodal(pnode=i, df = df)
            
            
    df_collection[0]['value'] = df_collection[0].iloc[:,0:total[0]].sum(axis=1)/total[0]
    df_collection[1]['value'] = df_collection[1].iloc[:,0:total[1]].sum(axis=1)/total[1]
    df_collection[2]['value'] = df_collection[2].iloc[:,0:total[2]].sum(axis=1)/total[2]
     
    '''
    group_0['value'] = group_0['value']/total[0]
    group_1['value'] = group_1['value']/total[1]
    group_2['value'] = group_2['value']/total[2]
    '''
     
    group_0.to_csv('8760_cluster_0.csv')
    group_1.to_csv('8760_cluster_1.csv')
    group_2.to_csv('8760_cluster_2.csv')