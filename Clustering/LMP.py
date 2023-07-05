# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 12:32:48 2021

@author: 54651
"""
import pandas as pd

df = pd.read_csv('pjm_dom_13KV_loadbus_rtlmps_may2020_april2021.csv')

columns = ['datetime_beginning_ept', 'pnode_id',
           'pnode_name', 'total_lmp_rt']

df = df[columns]

df['datetime_beginning_ept'] = pd.to_datetime(df['datetime_beginning_ept'])

# df.datetime_beginning_ept.max()
#Out[21]: Timestamp('2021-05-01 23:00:00')

df.datetime_beginning_ept.min()
#Out[22]: Timestamp('2020-05-01 00:00:00')

ids = df.pnode_id.unique()

#all_nodes = []


for i in ids:
    dummy = df[df.pnode_id == i]
    print(i, dummy.datetime_beginning_ept.min(), dummy.datetime_beginning_ept.max())

'''
    dummy.sort_values("datetime_beginning_ept", inplace=True)
    # dropping ALL duplicate values
    dummy.drop_duplicates(subset="datetime_beginning_ept",
                          keep=False, inplace=True)
    t = dummy['total_lmp_rt'].describe()
    cnt_over_mean = dummy[dummy['total_lmp_rt'] > t[1]].count()[0]
    results = [i, dummy.pnode_name.unique(), t[1], t[2], t[3], t[7], cnt_over_mean]
    all_nodes.append(results)
'''
