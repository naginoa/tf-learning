# 特征工程
# 1. 移除异常点
ulimit = np.percentile(data_train.price_doc.values, 99)
llimit = np.percentile(data_train.price_doc.values, 1)
data_train.loc[data_train['price_doc'] >ulimit, 'price_doc'] = ulimit
data_train.loc[data_train['price_doc'] <llimit, 'price_doc'] = llimit

# 2. 删除缺失值过半的特征
drop_columns = missing_df.ix[missing_df['missing_count']>0.5, 'column_name'].values
data_train.drop(drop_columns, axis=1, inplace=True)
data_test.drop(drop_columns, axis=1, inplace=True)

# 3. 删除不正常的行数据
data_all.drop(data_train[data_train["life_sq"] > 7000].index, inplace=True)

# 4. 提取时间
# week of year #
data_all["week_of_year"] = data_all["timestamp"].dt.weekofyear
# day of week #
data_all["day_of_week"] = data_all["timestamp"].dt.weekday
# yearmonth
data_all['yearmonth'] = pd.to_datetime(data_all['timestamp'])
data_all['yearmonth'] = data_all['yearmonth'].dt.year*100 + data_all['yearmonth'].dt.month
data_all_groupby = data_all.groupby('yearmonth')

# 5. 连续数据离散化
data_all['floor_25'] = (data_all['floor']>25.0)*1

# 6. 分组来填补平均值
for num in number_columns:
    if(sum(data_all[num].isnull())>0):
        isnull_raw = data_all[num].isnull()
        isnull_yearmonth = data_all.ix[isnull_raw, 'yearmonth'].values
        data_all_groupby[num].transform(lambda x: x.fillna(x.mean()))

# 7. Get_dummies离散化
dummies = pd.get_dummies(data=data_all[ob], prefix="{}#".format(ob))
data_all.drop(ob, axis=1, inplace=True)
data_all = data_all.join(dummies)

# 8. 用radio中位数填补空缺
kitch_ratio = train_df['full_sq']/train_df['kitch_sq']
train_df['kitch_sq']=train_df['kitch_sq'].fillna(train_df['full_sq'] /kitch_ratio.median())

# 9. LabelEncoder 
for ob in object_columns:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(data_train[ob].values))
    data_train[ob] = lbl.fit_transform(list(data_train[ob].values))

# 10. PCA的可视化与转换
from sklearn.decomposition import PCA
components = 20
model = PCA(n_components=components)
model.fit(data_train)
ex_variance = pd.DataFrame({'ex_variance':model.explained_variance_ratio_ [0:components], 'n_component':range(1,components+1)})
ax = sns.barplot(x='n_component', y='ex_variance', data=ex_variance)
ax.set_title('PCA_variance_explained')
plt.show()
data_train = model.fit_transform(data_train)
data_test = model.fit_transform(data_test)