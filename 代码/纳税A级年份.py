# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import pandas as pd
import numpy as np

data_input = pd.read_excel(r'E:\Graduate_Project\competition1\yyx\纳税A级年份.xlsx')
# data_output = pd.read_excel(r'E:\Graduate_Project\competition1\after\纳税A级年份_特征.xlsx')
print(data_input.head())

col_names = data_input.columns.tolist()
print(col_names)

col_names = col_names + ['纳税A级年份:2014', '纳税A级年份:2015', '纳税A级年份:2016', '纳税A级年份:2017']
print(col_names)
data_input = data_input.reindex(columns=col_names)
print(data_input.head())
print(data_input['纳税A级年份'])

for i in [2014, 2015, 2016, 2017]:
    data_input.loc[data_input['纳税A级年份']==i, '纳税A级年份:'+str(i)]=1
    data_input.loc[data_input['纳税A级年份']!=i, '纳税A级年份:'+str(i)]=0

del data_input['纳税A级年份']
data_output = data_input.groupby('企业编号').sum() #groupby函数分类汇总
data_output = data_output.reset_index() #reset_index函数将索引变成列。
# data_output = data_output.reindex(index = np.arange(data_input.shape[0]))

print(data_output.head())  


data_output.to_excel('E:/Graduate_Project/competition1/after/纳税A级年份_特征.xlsx', index = False)






