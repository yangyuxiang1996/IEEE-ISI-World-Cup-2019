import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler


data = pd.read_excel('data_0.xlsx')
X = data.drop(['企业总评分', '企业编号'], axis=1)
y = data['企业总评分']
target = '企业总评分'
ID = '企业编号'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
Z_normal = StandardScaler()
X_train = Z_normal.fit_transform(X_train)
X_test = Z_normal.transform(X_test)

kf = KFold(n_splits=5, shuffle=True, random_state=1)
xgb_model = xgb.XGBRegressor()


def modelfit(alg, dtrain, y_train, dtest, y_test, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=y_train.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='rmse', early_stopping_rounds=early_stopping_rounds, show_stdv=False)

    # fit the algorithm on the data
    alg.fit(dtrain[predictors], y_train, eval_metric='rmse')

    # predict the training set
    dtrain_prediction = alg.predict(dtrain[predictors])
    dtest_prediction = alg.predict(dtest[predictors])
    train_out = pd.DataFrame(list(zip(y_train, pd.Series(dtrain_prediction))), index=y_train.index, columns=['y_true', 'y_pred'])
    test_out = pd.DataFrame(list(zip(y_test, pd.Series(dtest_prediction))), index=y_test.index, columns=['y_true', 'y_pred'])

    # print model report
    print("\nModel Report")
    print("Train RMSE : %.4g" % mean_squared_error(y_train.values, dtrain_prediction)**0.5)
    print("Test RMSE : %.4g" % mean_squared_error(y_test.values, dtest_prediction) ** 0.5)

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importance')
    plt.ylabel('Feature Importance Score')
    plt.show()

    return train_out, test_out


# Choose all predictors except target & ID
features = [x for x in X.columns if x not in [target, ID]]
xgb1 = XGBRegressor(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    # objective='reg:squarederror',
    seed=1,
    tree_method='gpu_exact'
)
X_train = pd.DataFrame(X_train, columns=features)
X_test = pd.DataFrame(X_test, columns=features)
train_out, test_out = modelfit(xgb1, X_train, y_train, X_test, y_test, predictors=features)


