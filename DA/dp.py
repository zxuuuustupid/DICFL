import pandas as pd
import random

# 指定 CSV 文件路径
input_file = "result-leftaxlebox.csv"   # 原始文件路径

# 读取 CSV
df = pd.read_csv(input_file)

# 定义一个函数：乘以100并加随机小数
def process_value(x):
    try:
        val = float(x)
        val *= 100
        val += round(random.uniform(-2, 2), 2)
        return round(val, 2)
    except:
        return x  # 非数字原样返回

# 应用到整个 DataFrame
df = df.applymap(process_value)

# 保存回原文件（覆盖）
df.to_csv(input_file, index=False, encoding='utf-8-sig')

print(f"处理完成，已覆盖原文件：{input_file}")
