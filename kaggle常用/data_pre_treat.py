# 数据预处理
# 1. 读取数据：
data_macro = pd.read_csv("macro.csv", parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)

# 2. 显示为object的属性：
data_train.dtypes[data_train.dtypes=='object']

# 3. 改变数据类型
data_train['material'] = data_train['material'].astype('object')

# 4. 概览数据
data_train.describe(include=['object'])
# 5. 合并两个表（上下）
data_all = pd.concat([data_train, data_test], ignore_index=True)

# 6. 合并两个表（左右）
data_all = pd.merge(data_all, data_macro, on='timestamp', how='left')

# 7. 提取Number， Object特征：
object_columns =  data_all.columns[data_all.dtypes == 'object']
number_columns = data_all.columns[data_all.dtypes != 'object']

# 8. 计算两个特征平均
sa_price = train_df.groupby('sub_area')[['work_share', 'price_doc']].mean()