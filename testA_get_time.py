import pandas as pd
import csv
import numpy as np

# 读取xlsx文件
df = pd.read_excel('测试集A.xlsx')

# 提取指定列的前8位值
column_values = df[['220X75', '310X56']].astype(str).apply(lambda x: x.str[:8])

# 保存为csv文件
column_values.to_csv('testA_get_time.csv', index=False)

# 创建800x800的二维数组，初始值全部为0
array = np.zeros((300, 300), dtype=int)

# 从CSV文件中读取第一列数据
with open('testA_get_time.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # 跳过第一行
    column_data = [int(row[0]) for row in reader]

# 循环处理列数据
for i, num1 in enumerate(column_data):
    # 将该列所有值减去第一个数值
    adjusted_column = [num - num1 for num in column_data]

    # 根据要求更新二维数组的值
    for j, num2 in enumerate(adjusted_column):
        if num2 == 0 or num2 == -1:
            array[i, j] = 1

# 将二维数组保存到CSV文件
np.savetxt('testA_adj.csv', array, delimiter=',', fmt='%d')