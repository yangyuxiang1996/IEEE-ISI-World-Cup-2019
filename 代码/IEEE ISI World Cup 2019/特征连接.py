# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import os
# import matplotlib.pyplot as plt


def merge_features(path):
    data = pd.read_excel(path + '赛题1结果_团队名.xlsx')
    # data = label
    const_xlsx_file = ".xlsx"
    for filename in os.listdir(path):
        if filename == '赛题1结果_团队名.xlsx' or filename == '特征总和.xlsx':
            continue
        if os.path.splitext(filename)[1] == const_xlsx_file:
            feature = pd.read_excel(path+filename)
            print(filename)
            print(feature.head())
            data = pd.merge(data, feature, how='left')
            print(data.head())

    return data


data = merge_features("E:\\Nutstore\\IEEE ISI World Cup 2019\\测试集特征文件\\")
data.to_excel('test.xlsx', index=False)

