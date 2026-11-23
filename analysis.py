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
for i in range(1,4):
    m=mm[mm['Pclass']==i].groupby(['CabinGroup']).agg(
        生还率=("Survived","mean"),
        总人数=("PassengerId",'count'),
        平均年龄=("Age",'mean'),
    )
    print(m)

#------------ 男女生存率差异大都原因分析
m=s.groupby('Sex').agg(
    平均年龄=('Age','mean'),
    中位数=('Age','median'),
    最大年龄=('Age','max'),
)
print(m)
print('--------------------------------')
m=s.groupby(['Sex','Pclass']).size();
print(m)
print('----------------------------')
print(m)
m=s.groupby('Pclass')['Survived'].mean()
print(m)
m=mm.groupby(['Sex','CabinGroup']).size()
print(m)


#--------------------清洗数据----------------
s2=s.copy()
#缺失值
s['Age']=s['Age'].fillna(s['Age'].median())
print(s['Age'].isna().sum())
print('-----------------------------')

s['HasCabin']=s['Cabin'].notna()
m1=s.groupby('HasCabin')['Survived'].mean()
print(m1)
m2=s.groupby(['HasCabin','Sex'])['Survived'].mean()
print(m2)
s['Embarked']=s['Embarked'].fillna(s['Embarked'].mode()[0])

#重复值
print("完全重复行:",s.duplicated().sum())
print("姓名重复: ",s['Name'].duplicated().sum())

#矛盾情况
print(s[(s['Fare']<10) & (s['Pclass']==1)])

#---------------验证