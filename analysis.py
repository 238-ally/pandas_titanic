#打印出表格前7行
import pandas as pd

pd.set_option("display.max.columns",None)
pd.set_option("display.width",200)
s=pd.read_csv("titanic.csv")
print(s.head())
print("-------------------------------")
print(s.shape)
print('---------------------------')
print(s.info())
print("-------------------------")
print(s["Survived"].mean())
print("-----------------------")
print(s['Sex'].value_counts())
print("-------------------")
print(s['Age'].describe())