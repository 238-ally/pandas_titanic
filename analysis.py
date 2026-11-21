#打印出表格前7行
import pandas as pd

#------------显示列数设置--------------
pd.set_option("display.max.columns",5)
pd.set_option("display.width",200)

#------------------读数据-------------
s=pd.read_csv("titanic.csv")

#--------------描述统计--------------
print(s.head())
print("---------------------------")
print(s.shape)
print('---------------------------')
print(s.info())
print("---------------------------")
print(s["Survived"].mean())
print("---------------------------")
print(s['Sex'].value_counts())
print("---------------------------")
print(s['Age'].describe())

print("****************************")
#-------生还率与性别,舱位,年龄有什么关系吗--------
m=s.groupby("Sex")["Survived"].mean()
print(m)
print("----------------------------")
m=s.groupby("Age")["Survived"].mean()
print(m)
print("----------------------------")
s["AgeGroup"]=pd.cut(s['Age'],bins=[0,20,25,30,40,55,80],labels=["婴儿","少年",'青年','中年','中老年','老年'])
m=s.groupby("AgeGroup")["Survived"].mean()
print(m)
print("----------------------------")
mm=s.dropna(subset=["Cabin"]).copy()
mm["CabinGroup"]=mm['Cabin'].str[0]
m=mm.groupby("CabinGroup")['Survived'].mean()
print(m)
print("----------------------------")
m=s.groupby("Sex")['Survived'].mean()
print(m)
#女性生存率远高于男性,BDEF仓存活率最高,T仓生还概率为0
#查看T仓原本有多少乘客
m=mm[mm['CabinGroup'].str[0]=='T']
print(m)
print("-----------------------------")
print(m[['PassengerId','Sex','Age']])
print('------------------------------------')
#---女性生存率高与仓位,年龄,性别有关系吗--------
m=mm.groupby(['Sex','CabinGroup'])['Survived'].mean()
print(m)
print('------------------------------------')
m=s.groupby(['AgeGroup','Sex'])['Survived'].mean()
print(m)
print('---------------------------------***')
#------------甲板,舱位对生还率影响的占比'
#1等舱位分析各甲板对生还率的影响
#print(mm.info())
for i in range(1,4):
    m=mm[mm['Pclass']==i].groupby(['CabinGroup']).agg(
        生还率=("Survived","mean"),
        总人数=("PassengerId",'count'),
        平均年龄=("Age",'mean'),
    )
    print(m)