import pandas as pd
import numpy as np
import xgboost as xgb
import visuals as vs
from time import time
from xgboost import plot_importance
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel


data_train = pd.read_excel("train.xlsx")
data_test = pd.read_excel("test.xlsx")

features = data_test.columns.tolist()
data_score = data_train[['企业编号', '企业总评分']]
data_train = data_train[features]

data_train.drop_duplicates(inplace=True)
data_score.drop_duplicates(inplace=True)

limit_percent = 0.25
limit_value = len(data_train) * limit_percent
print(list(data_train.columns[data_train.isna().sum() > limit_value]))

data_train.drop('扣非净利润滚动环比增长', axis=1, inplace=True)
data_test.drop('扣非净利润滚动环比增长', axis=1, inplace=True)

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

# xgboost
X = data_train.drop('企业编号', axis=1)
y = data_score.drop('企业编号', axis=1)
target = '企业总评分'
ID = '企业编号'
features_name = X.columns.tolist()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_val = min_max_scaler.transform(X_val)
X_train = pd.DataFrame(X_train, columns=features_name)
X_val = pd.DataFrame(X_val, columns=features_name)

kf = KFold(n_splits=5, random_state=1)
xgb_model = xgb.XGBRegressor()


def modelfit(alg, dtrain, y_train, dtest, y_test, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_parameters = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain.values, label=y_train.values)
        cvresult = xgb.cv(xgb_parameters, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='rmse', early_stopping_rounds=early_stopping_rounds, show_stdv=False)
        print(cvresult)
        alg.set_params(n_estimators=cvresult.shape[0])

    alg.fit(dtrain, y_train, eval_metric='rmse')

    dtrain_prediction = alg.predict(dtrain)
    dtest_prediction = alg.predict(dtest)

    # print model report
    print("\nModel Report")
    print("Train RMSE : %.4g" % mean_squared_error(y_train.values, dtrain_prediction)**0.5)
    print("Test RMSE : %.4g" % mean_squared_error(y_test.values, dtest_prediction) ** 0.5)

    # feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    # feat_imp.plot(kind='bar', title='Feature Importance')
    # plt.ylabel('Feature Importance Score')
    # plt.show()
    plot_importance(alg)
    plt.show()

    importances = alg.feature_importances_
    vs.feature_plot(importances, dtrain, y_train)

    return dtrain_prediction, dtest_prediction

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
    tree_method='gpu_exact'
)

start = time()
dtrain_prediction, dtest_prediction = modelfit(xgb1, X_train, y_train, X_val, y_val)
end = time()
train_out = pd.DataFrame(list(zip(y_train.values.flatten(), pd.Series(dtrain_prediction))),
                         index=y_train.index, columns=['y_true', 'y_pred'])
test_out = pd.DataFrame(list(zip(y_val.values.flatten(), pd.Series(dtest_prediction))),
                        index=y_val.index, columns=['y_true', 'y_pred'])

print("the model fit time: ", end-start)

# # TODO: Tune max_depth and min_child_weight
# param_test1 = {'max_depth': range(3, 10, 1), 'min_child_weight': range(1, 6, 2)}
# gsearch1 = GridSearchCV(estimator=xgb1, param_grid=param_test1, scoring=make_scorer(mean_squared_error),
#                         iid=False, cv=5)
# grid_obj = gsearch1.fit(X_train, y_train)
# print(grid_obj.best_params_)
# print(grid_obj.best_score_)

# TODO: Tune the gamma
xgb2 = XGBRegressor(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=8,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    # objective='reg:squarederror',
    seed=10,
    # nthread=4,
    # tree_method='gpu_exact'
)
# param_test2 = {'gamma': [i/10.0 for i in range(11)]}
# gsearch2 = GridSearchCV(estimator=xgb1, param_grid=param_test2, scoring=make_scorer(mean_squared_error),
#                         iid=False, cv=5)
# grid_obj2 = gsearch2.fit(X_train, y_train)
# print(grid_obj2.best_params_)
# print(grid_obj2.best_score_)

xgb3 = XGBRegressor(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=8,
    min_child_weight=1,
    gamma=0.5,
    subsample=0.8,
    colsample_bytree=0.8,
    # objective='reg:squarederror',
    seed=10,
    # nthread=4,
    # tree_method='gpu_exact'
)

# TODO: tune the subsample and colsample_bytree
# param_test3 = {'subsample': [0.5, 0.6, 0.7],
#                'colsample_bytree': [0.5, 0.6, 0.7]}
# gsearch3 = GridSearchCV(estimator=xgb3, param_grid=param_test3, scoring=make_scorer(mean_squared_error),
#                         iid=False, cv=5)
# grid_obj3 = gsearch3.fit(X_train, y_train)
# print(grid_obj3.best_params_)
# print(grid_obj3.best_score_)

xgb4 = XGBRegressor(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=8,
    min_child_weight=1,
    gamma=0.5,
    subsample=0.8,
    colsample_bytree=0.8,
    # objective='reg:squarederror',
    seed=10,
    # nthread=4,
    # tree_method='gpu_exact'
)
# TODO: tune the Regularization Parameters
# param_test4 = {'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]}
# gsearch4 = GridSearchCV(estimator=xgb4, param_grid=param_test4, scoring=make_scorer(mean_squared_error),
#                         iid=False, cv=5)
# grid_obj4 = gsearch4.fit(X_train, y_train)
# print(grid_obj4.best_params_)
# print(grid_obj4.best_score_)

xgb5 = XGBRegressor(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=8,
    min_child_weight=1,
    gamma=0.6,
    subsample=0.8,
    colsample_bytree=0.8,
    # objective='reg:squarederror',
    seed=10,
    reg_alpha=1,
    # nthread=4,
    # tree_method='gpu_exact'
)
start = time()
dtrain_prediction, dtest_prediction = modelfit(xgb5, X_train, y_train, X_val, y_val)
end = time()
train_out = pd.DataFrame(list(zip(y_train.values.flatten(), pd.Series(dtrain_prediction))),
                         index=y_train.index, columns=['y_true', 'y_pred'])
test_out = pd.DataFrame(list(zip(y_val.values.flatten(), pd.Series(dtest_prediction))),
                        index=y_val.index, columns=['y_true', 'y_pred'])
print("the model fit time: ", end-start)

# TODO: feature selection
print("starting feature selection")
# thresholds = np.sort(xgb5.feature_importances_)
# for thresh in thresholds:
#     selection = SelectFromModel(xgb5, threshold=thresh, prefit=True)
#     selection_X_train = selection.transform(X_train)
#
#     xgb5.fit(selection_X_train, y_train.values)
#
#     selection_X_val = selection.transform(X_val)
#     y_train_pred = xgb5.predict(selection_X_train)
#     y_val_pred = xgb5.predict(selection_X_val)
#
#     train_rmse = mean_squared_error(y_train, y_train_pred) ** 0.5
#     val_rmse = mean_squared_error(y_val, y_val_pred) ** 0.5
#     print("thresh=%.3f, n=%d, train rmse=%.4f, val rmse=%.4f" %
#           (thresh, selection_X_train.shape[1], train_rmse, val_rmse))
model = SelectFromModel(xgb1, prefit=True)
selection_X_train = model.transform(X_train)
selection_X_val = model.transform(X_val)

xgb5.fit(selection_X_train, y_train)
y_train_pred = xgb5.predict(selection_X_train)
y_val_pred = xgb5.predict(selection_X_val)

train_rmse = mean_squared_error(y_train, y_train_pred) ** 0.5
val_rmse = mean_squared_error(y_val, y_val_pred) ** 0.5
print(" n=%d, train rmse=%.4f, val rmse=%.4f" % (selection_X_train.shape[1], train_rmse, val_rmse))

#
xgb6 = XGBRegressor()
xgb6.fit(X_train, y_train)
y_train_pred = xgb5.predict(X_train)
y_val_pred = xgb5.predict(X_val)

train_rmse = mean_squared_error(y_train, y_train_pred) ** 0.5
val_rmse = mean_squared_error(y_val, y_val_pred) ** 0.5
print(" n=%d, train rmse=%.4f, val rmse=%.4f" % (selection_X_train.shape[1], train_rmse, val_rmse))

