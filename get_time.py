import csv
import pandas as pd

# 打开原始csv文件
with open('time_output.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader)  # 保存csv文件中的所有行数据

# 对每列除第一行外的值进行处理
for i in range(1, len(data)):
    for j in range(len(data[i])):
        if data[i][j] != '':
            data[i][j] = int(data[i][j][:8])  

# 将处理后的数据写入新的csv文件
with open('get_time.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)


# 读取csv文件，跳过第一行表头，并不使用列头
df = pd.read_csv('get_time.csv',index_col=0)

# 删除重复的列
df = df.T.drop_duplicates().T


# 将删除重复列后的数据重新保存为 CSV 文件
df.to_csv('get_time_delete.csv', index=False)

df = pd.read_csv('get_time_delete.csv')

# 指定要删除的列名
columns_to_delete = ['220X71', '220X151']

# 删除指定的列
df = df.drop(columns_to_delete, axis=1)
df = df.astype(int)

# 保存结果到新的CSV文件
df.to_csv('get_time_delete_end.csv', index=False)