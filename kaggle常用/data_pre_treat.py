# ����Ԥ����
# 1. ��ȡ���ݣ�
data_macro = pd.read_csv("macro.csv", parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)

# 2. ��ʾΪobject�����ԣ�
data_train.dtypes[data_train.dtypes=='object']

# 3. �ı���������
data_train['material'] = data_train['material'].astype('object')

# 4. ��������
data_train.describe(include=['object'])
# 5. �ϲ����������£�
data_all = pd.concat([data_train, data_test], ignore_index=True)

# 6. �ϲ����������ң�
data_all = pd.merge(data_all, data_macro, on='timestamp', how='left')

# 7. ��ȡNumber�� Object������
object_columns =  data_all.columns[data_all.dtypes == 'object']
number_columns = data_all.columns[data_all.dtypes != 'object']

# 8. ������������ƽ��
sa_price = train_df.groupby('sub_area')[['work_share', 'price_doc']].mean()