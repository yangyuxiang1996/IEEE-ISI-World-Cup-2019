import pandas as pd
import numpy as np
import xgboost as xgb
from time import time
from xgboost import plot_importance
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor


# step1: 文件输入：读取数据
data_train = pd.read_excel("train.xlsx")
data_test = pd.read_excel("test.xlsx")

features = data_test.columns.tolist()
data_score = data_train[['企业编号', '企业总评分']]
data_train = data_train[features]

# step2: 数据预处理
# 2.1：去重
data_train.drop_duplicates(inplace=True)
data_score.drop_duplicates(inplace=True)

# 2.2：去除较多空值的特征
limit_percent = 0.25
limit_value = len(data_train) * limit_percent
print(list(data_train.columns[data_train.isna().sum() > limit_value]))

data_train.drop('扣非净利润滚动环比增长', axis=1, inplace=True)
data_test.drop('扣非净利润滚动环比增长', axis=1, inplace=True)

# 2.3：填补空值
data_train_missing_columns = list(data_train.columns[data_train.isna().sum() != 0])
print(data_train_missing_columns)
data_test_missing_columns = list(data_test.columns[data_test.isna().sum() != 0])
print(data_test_missing_columns)

fill_median = ['利润总额增长率', '加权利润总额', '加权营业成本', '加权销售费用', '营业成本增长率', '销售成本增长率',
               '加权扣非净利润同比增长', '加权营业总收入滚动环比增长', '加权营收总收入同比增长',
               '加权投资现金流量净额', '加权筹资现金流量净额', '投资现金流量净额增长率', '筹资现金流量净额增长率', '加权流动比率',
               '加权速动比率', '流动比率增长率', '速动比率增长率', '净资产收益率增长率', '加权净资产收益率', '加权毛利率',
               '毛利率增长率', '产权比率增长率', '加权产权比率', '加权负债与有形净资产比率', '负债与有形净资产比率增长率',
               '加权存货周转天数', '加权应收账款周转天数', '加权总资产周转率', '存货周转天数增长率', '应收账款周转天数增长率',
               '总资产周转率增长率', '加权每股净资产', '加权每股未分配利润', '加权每股经营现金流', '每股净资产增长率',
               '每股未分配利润增长率', '每股经营现金流增长率', '城镇职工基本养老保险人数', '实缴出资额']
fill_0 = ['示意图', '电影', '设计图', '建筑', '美术', '模型', '音乐',
          '曲艺', '舞蹈', '摄影', '类似摄影作品', '图形', '文字', '缺失',
          '纳税A级年份:2014', '纳税A级年份:2015', '纳税A级年份:2016', '纳税A级年份:2017']

data_train[fill_median] = data_train[fill_median].fillna(data_train[fill_median].median())
data_test[fill_median] = data_test[fill_median].fillna(data_test[fill_median].median())
data_train[fill_0] = data_train[fill_0].fillna(0)
data_test[fill_0] = data_test[fill_0].fillna(0)

# step3：核心算法：xgboost
X = data_train.drop('企业编号', axis=1)
y = data_score.drop('企业编号', axis=1)
X_test = data_test.drop('企业编号', axis=1)
target = '企业总评分'
ID = '企业编号'
features_name = X.columns.tolist()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_val = min_max_scaler.transform(X_val)
X_test = min_max_scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=features_name)
X_val = pd.DataFrame(X_val, columns=features_name)
X_test = pd.DataFrame(X_test, columns=features_name)

# modelfit函数
def modelfit(alg, dtrain, y_train, dtest, y_test, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_parameters = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain.values, label=y_train.values)
        cvresult = xgb.cv(xgb_parameters, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='rmse', early_stopping_rounds=early_stopping_rounds, show_stdv=False)
        print("the cv number is: ", cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])

    alg.fit(dtrain, y_train, eval_metric='rmse')

    dtrain_prediction = alg.predict(dtrain)
    dtest_prediction = alg.predict(dtest)

    # print model report
    print("\nModel Report")
    print("feature numbers: %d" % dtrain.shape[1])
    print("Train RMSE : %.4g" % mean_squared_error(y_train.values, dtrain_prediction)**0.5)
    print("Test RMSE : %.4g" % mean_squared_error(y_test.values, dtest_prediction) ** 0.5)

    # feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    # feat_imp.plot(kind='bar', title='Feature Importance')
    # plt.ylabel('Feature Importance Score')
    # plt.show()
    plot_importance(alg)
    plt.show()

    importances = alg.feature_importances_

    return dtrain_prediction, dtest_prediction


# 初始化模型
xgb0 = XGBRegressor(random_state=10, importance_type='gain')
start = time()
dtrain_prediction, dtest_prediction = modelfit(xgb0, X_train, y_train, X_val, y_val)
end = time()
print("the model fit time: %.4f" % (end-start))

train_out = pd.DataFrame(list(zip(y_train.values.flatten(), pd.Series(dtrain_prediction))),
                         index=y_train.index, columns=['y_true', 'y_pred'])
test_out = pd.DataFrame(list(zip(y_val.values.flatten(), pd.Series(dtest_prediction))),
                        index=y_val.index, columns=['y_true', 'y_pred'])


# step4：特征选择
model = SelectFromModel(xgb0, prefit=True)
selection_X_train = model.transform(X_train)
selection_X_val = model.transform(X_val)
selection_X_test = model.transform(X_test)

start = time()
xgb0.fit(selection_X_train, y_train)
end = time()
y_train_pred = xgb0.predict(selection_X_train)
y_val_pred = xgb0.predict(selection_X_val)

train_rmse = mean_squared_error(y_train, y_train_pred) ** 0.5
val_rmse = mean_squared_error(y_val, y_val_pred) ** 0.5
print("After feature selection： n=%d, train rmse=%.4f, val rmse=%.4f, the model fit time: %.4f" %
      (selection_X_train.shape[1], train_rmse, val_rmse, (end-start)))


# step5: 参数调优

# 5.1：手动输入参数
xgb1 = XGBRegressor(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    # objective='reg:squarederror',
    seed=10,
    # nthread=4,
    tree_method='exact',
    random_state=10,
    importance_type='gain'
)

start = time()
dtrain_prediction, dtest_prediction = modelfit(xgb1, pd.DataFrame(selection_X_train),
                                               y_train, pd.DataFrame(selection_X_val), y_val)
end = time()
print("the model fit time: %.4f" % (end-start))
train_out = pd.DataFrame(list(zip(y_train.values.flatten(), pd.Series(dtrain_prediction))),
                         index=y_train.index, columns=['y_true', 'y_pred'])
test_out = pd.DataFrame(list(zip(y_val.values.flatten(), pd.Series(dtest_prediction))),
                        index=y_val.index, columns=['y_true', 'y_pred'])


# # 5.2 TODO: Tune max_depth and min_child_weight
print("\n--------------------------------------")
print("Tune max_depth and min_child_weight...")
cv = KFold(n_splits=5, random_state=10, shuffle=True)
param_test1 = {'max_depth': range(3, 11, 1), 'min_child_weight': range(1, 6, 1)}
gsearch1 = GridSearchCV(estimator=xgb1, param_grid=param_test1,
                        scoring=make_scorer(mean_squared_error, greater_is_better=False), iid=False, cv=cv)
grid_obj = gsearch1.fit(selection_X_train, y_train)
print(grid_obj.best_params_)
print(grid_obj.best_score_)
for i in range(len(grid_obj.cv_results_['params'])):
    print("the mean test score of {} is {}.".format(grid_obj.cv_results_['params'][i], grid_obj.cv_results_['mean_test_score'][i]))


# 5.3：TODO: tune the gamma
print("\n--------------------------------------")
print("Tune the gamma...")
xgb2 = grid_obj.best_estimator_
param_test2 = {'gamma': [i/100.0 for i in range(11)]}
gsearch2 = GridSearchCV(estimator=xgb2, param_grid=param_test2,
                        scoring=make_scorer(mean_squared_error, greater_is_better=False), iid=False, cv=cv)
grid_obj2 = gsearch2.fit(selection_X_train, y_train)
print(grid_obj2.best_params_)
print(grid_obj2.best_score_)
for i in range(len(grid_obj2.cv_results_['params'])):
    print("the mean test score of {} is {}.".format(grid_obj2.cv_results_['params'][i], grid_obj2.cv_results_['mean_test_score'][i]))

# 5.4：TODO: tune the subsample and colsample_bytree
print("\n--------------------------------------")
print("Tune the subsample and colsample_bytree...")
xgb3 = grid_obj2.best_estimator_
param_test3 = {'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
gsearch3 = GridSearchCV(estimator=xgb3, param_grid=param_test3,
                        scoring=make_scorer(mean_squared_error, greater_is_better=False), iid=False, cv=cv)
grid_obj3 = gsearch3.fit(selection_X_train, y_train)
print(grid_obj3.best_params_)
print(grid_obj3.best_score_)
for i in range(len(grid_obj3.cv_results_['params'])):
    print("the mean test score of {} is {}.".format(grid_obj3.cv_results_['params'][i], grid_obj3.cv_results_['mean_test_score'][i]))

# 5.5：TODO: tune the Regularization Parameters
print("\n--------------------------------------")
print("Tune the Regularization Parameters")
xgb4 = grid_obj3.best_estimator_
param_test4 = {'reg_lambda': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]}
gsearch4 = GridSearchCV(estimator=xgb4, param_grid=param_test4,
                        scoring=make_scorer(mean_squared_error, greater_is_better=False),
                        iid=False, cv=cv)
grid_obj4 = gsearch4.fit(selection_X_train, y_train)
print(grid_obj4.best_params_)
print(grid_obj4.best_score_)
for i in range(len(grid_obj4.cv_results_['params'])):
    print("the mean test score of {} is {}.".format(grid_obj4.cv_results_['params'][i], grid_obj4.cv_results_['mean_test_score'][i]))

# 5.6：predict the best model
print("\n--------------------------------------")
print("predicting the best model")
best_xgb = grid_obj4.best_estimator_
start = time()
best_xgb.fit(selection_X_train, y_train)
y_train_pred = best_xgb.predict(selection_X_train)
y_val_pred = best_xgb.predict(selection_X_val)
end = time()
print("the model fit time: %.4f" % (end-start))
train_rmse = mean_squared_error(y_train, y_train_pred) ** 0.5
val_rmse = mean_squared_error(y_val, y_val_pred) ** 0.5
print("the best model report： train rmse=%.4f, val rmse=%.4f, the model fit time: %.4f" %
      (train_rmse, val_rmse, (end-start)))

# step6：预测输出
y_test_pred = best_xgb.predict(selection_X_test)
result_ID = data_test['企业编号']
result_score = pd.Series(y_test_pred).apply(lambda x: int(round(x)))
result = pd.DataFrame({'ID': result_ID, 'score': result_score})
result.to_csv('mission1_YML.csv', index=False, header=False)


