import pandas as pd


# ETL
'''
filenames = ['rt_hrl_lmps_may_june_2020.csv', 'rt_hrl_lmps_jul_sep_2020.csv',
             'rt_hrl_lmps_sep_oct_2020.csv', 'rt_hrl_lmps_nov_dec_2020.csv',
             'rt_hrl_lmps_Jan_feb_2021.csv', 'rt_hrl_lmps_mar_apr_2021.csv']

df0 = pd.read_csv(filenames[0])
df1 = pd.read_csv(filenames[1])
df2 = pd.read_csv(filenames[2])
df3 = pd.read_csv(filenames[3])
df4 = pd.read_csv(filenames[4])
df5 = pd.read_csv(filenames[5])

result1 = df0.append(df1)
result1 = result1.append(df2)
result1 = result1.append(df3)
result1 = result1.append(df4)
result1 = result1.append(df5)

result1.to_csv('pjm_dom_13KV_loadbus_rtlmps_may2020_april2021.csv')
'''


# Analysis

df = pd.read_csv('pjm_dom_13KV_loadbus_rtlmps_may2020_april2021.csv')
df = df.iloc[:, 1:]
df = df[df.type == 'LOAD']
columns = ['datetime_beginning_ept', 'pnode_id',
           'pnode_name', 'total_lmp_rt']

df = df[columns]

df['datetime_beginning_ept'] = pd.to_datetime(df['datetime_beginning_ept'])

ids = df.pnode_id.unique()
all_nodes = []

for i in ids:
    dummy = df[df.pnode_id == i]
    if dummy.shape[0] > 8760:
        #dummy.sort_values("datetime_beginning_ept", inplace=True)
        # dropping ALL duplicate values
        dummy.drop_duplicates(subset="datetime_beginning_ept",
                              keep=False, inplace=True)
        dummy.index = dummy.datetime_beginning_ept
        dummy = dummy.resample('1H', base=8).mean()
        dummy = dummy.bfill()
        t = dummy['total_lmp_rt'].describe()
        cnt_over_mean = dummy[dummy['total_lmp_rt'] > t[1]].count()[0]
        results = [i, t[1], t[2], t[3],
                   t[7], dummy.skew()[1], dummy.kurtosis()[1], cnt_over_mean]
        all_nodes.append(results)

data = pd.DataFrame(all_nodes, columns=[
                    'pnode_id', 'mean', 'standard_deviation', 'min', 'max', 'skew', 'kurtosis', 'count_over_mean'])
data.to_csv('summary.csv')

# -------------------------------

df.datetime_beginning_ept.max()
#Out[21]: Timestamp('2021-05-01 23:00:00')

df.datetime_beginning_ept.min()
#Out[22]: Timestamp('2020-05-01 00:00:00')

ids = df.pnode_id.unique()

all_nodes = []


for i in ids:
    dummy = df[df.pnode_id == i]
    if str(dummy.datetime_beginning_ept.max()) != '2021-05-01 23:00:00':
        print(i, dummy.datetime_beginning_ept.max())

for i in ids:
    dummy = df[df.pnode_id == i]
    if dummy.shape[0] < 8760:
        print(i)
'''
#output:
34885301
76427737
1388604896
1666105931
2156111266
'''

'''
is_NaN = dummy.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = dummy[row_has_NaN]
print(rows_with_NaN)
'''

for i in ids:
    dummy = df[df.pnode_id == i]
    dummy.sort_values("datetime_beginning_ept", inplace=True)
    # dropping ALL duplicate values
    dummy.drop_duplicates(subset="datetime_beginning_ept",
                          keep=False, inplace=True)
    print(i, '--->', dummy.shape[0])


for i in ids:
    dummy = df[df.pnode_id == i]
    dummy.sort_values("datetime_beginning_ept", inplace=True)
    # dropping ALL duplicate values
    dummy.drop_duplicates(subset="datetime_beginning_ept",
                          keep=False, inplace=True)
    t = dummy['total_lmp_rt'].describe()
    cnt_over_mean = dummy[dummy['total_lmp_rt'] > t[1]].count()[0]
    results = [i, dummy.pnode_name.unique(), t[1], t[2], t[3], t[7], cnt_over_mean]
    all_nodes.append(results)
