# 创建模型
# 1. import xgboost as xgb
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.grid_search import GridSearchCV
train_X = data_train
test_X = data_test
dtrain = xgb.DMatrix(train_X, train_y)
xgb_params={ 'eta': 0.05,
             'max_depth': 5,
             'subsample': 0.7,
             'colsample_bytree': 0.7,
             'objective': 'reg:linear',
             'eval_metric': 'rmse',
             'silent': 1
}

cv_output  = xgb.cv(dict(xgb_params, silent=0), dtrain, num_boost_round=1000, early_stopping_rounds=20, verbose_eval=20)
num_boost_round = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=320)
num_boost_round = model.best_iteration
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_round)
preds = np.exp(model.predict(xgb.DMatrix(test_X, feature_names=test_X.columns.values)))-1
submission = pd.DataFrame()
submission['id'] = test_id
submission["price_doc"]= preds

submission.to_csv("sub.csv",index=False)

# 画feature_importance
%matplotlib inline
fig, ax = plt.subplots(1, 1, figsize=(8, 60))
xgb.plot_importance(model, height=0.5, ax=ax)

# 提取feature_importance
import operator 
importance = model.get_fscore()

df_importance = pd.DataFrame(importance, columns=['feature', 'fscore'])
df_importance.sort_values(ascending=False)