import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt

"""
    对flow_model2_eval.py的结果用可视化显示
"""

sns.set(style="darkgrid")
# 获取数据
# titanic = sns.load_dataset("titanic") #who
titanic = pd.read_csv('result_pred_0.txt')  # 正样本
# titanic = titanic[titanic['probability'] > 0.5]
# titanic = titanic[(titanic['probability'] < 0.8) & (titanic['probability'] > 0.5)]
# titanic = pd.read_csv('D:\\Windows 10 Documents\\Downloads\\result_pred_0.txt')  # 负样本
titanic['probability'] = round((titanic['probability']), 2)
print(titanic['probability'].value_counts(bins=50))
"""
案例1：显示单个分类变量的值统计数
"""
# sns.countplot(x="probability", data=titanic)
# plt.show()