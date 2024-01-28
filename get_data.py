import pandas as pd

# 读取原始xlsx文件
df = pd.read_excel('dataset_train.xlsx')

# 初始化一个空列表，用于存储均值大于2017060000000000的列
selected_columns = []

# 遍历每一列
for column in df.columns:
    # 跳过字符列
    if not pd.api.types.is_numeric_dtype(df[column]):
        continue

    # 跳过第一行
    column_mean = df[column].iloc[1:].mean()
    #column_mean = df[column].iloc[1:, :].mean()

    # 判断均值是否大于2017060000000000
    if column_mean > 2017060000000000:
        selected_columns.append(column)

# 保存到新的CSV文件
df[selected_columns].to_csv('time_output.csv', index=False)

