from function import *

###########
# 聚类分析 #
###########
# 重新载入内容
TrainSet = pandas.read_csv(open('test.csv'))
TestSet = pandas.read_csv(open('train.csv'))

RedundantAttribute = ['PassengerId', 'Name', 'Ticket']
NominalAttribute = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Ticket']
NumericAttribute = ['SibSp', 'Age', 'Parch', 'Fare']

# 把属性中的字符串值改为数值
to_encode_attr = ['Embarked', 'Cabin', 'Ticket'];
TrainSet = encode_target(TrainSet, to_encode_attr)

# Sex属性male转为1，female转为0
TrainSet['Sex'].replace(to_replace='male', value=1, inplace=True)
TrainSet['Sex'].replace(to_replace='female', value=0, inplace=True)
TestSet['Sex'].replace(to_replace='male', value=1, inplace=True)
TestSet['Sex'].replace(to_replace='female', value=0, inplace=True)

# Age属性nan填充均值
TrainSet['Age'] = TrainSet['Age'].fillna(value=TrainSet['Age'].mean())
TestSet['Age'] = TestSet['Age'].fillna(value=TestSet['Age'].mean())

# Fare属性nan填充均值
TestSet['Fare'] = TestSet['Fare'].fillna(value=TestSet['Fare'].mean())

# 丢弃多余属性列
TrainSet = TrainSet.drop(columns=RedundantAttribute)
TestSet = TestSet.drop(columns=RedundantAttribute)

# k均值聚类
kmeans_cluster = KMeans(n_clusters=4, random_state=0).fit(TrainSet)

# DBSCAN聚类
dbscan_cluster = DBSCAN(eps=10, min_samples=5).fit(TrainSet)

# 显示聚类结果
pyplot.figure()

# kmeans聚类结果显示
temp = pandas.concat([DataFrame(kmeans_cluster.labels_, columns=['ClassLabel']), TrainSet], axis=1)
type1 = temp.loc[temp['ClassLabel'] == 0]
type2 = temp.loc[temp['ClassLabel'] == 1]
type3 = temp.loc[temp['ClassLabel'] == 2]
type4 = temp.loc[temp['ClassLabel'] == 3]

figure1 = pyplot.subplot(2, 1, 1)
pyplot.rcParams['font.sans-serif'] = ['SimHei']
figure1.set_title(u'k-means聚类结果')

type1_distribute = figure1.scatter(type1['Age'], type1['Fare'], c='green', marker='d')
type2_distribute = figure1.scatter(type2['Age'], type2['Fare'], c='red', marker='*')
type3_distribute = figure1.scatter(type3['Age'], type3['Fare'], c='black', marker='p')
type4_distribute = figure1.scatter(type4['Age'], type4['Fare'], c='brown')
pyplot.xlabel(u'某类人的年龄')
pyplot.ylabel(u'某类人的票价')
figure1.legend((type1_distribute, type2_distribute, type3_distribute, type4_distribute), ('第一类', '第二类', '第三类', '第四类'),
               loc=1)

# DBSCAN聚类结果显示
temp = pandas.concat([DataFrame(dbscan_cluster.labels_, columns=['ClassLabel']), TrainSet], axis=1)
temp = temp.drop(temp[temp['ClassLabel'] == -1].index)
print('DBSCAN聚类个数:', len(temp['ClassLabel'].unique()), 'DBSCAN类标', temp['ClassLabel'].unique())
type1 = temp.loc[temp['ClassLabel'] == 0]
type2 = temp.loc[temp['ClassLabel'] == 1]
type3 = temp.loc[temp['ClassLabel'] == 2]
type4 = temp.loc[temp['ClassLabel'] == 3]

figure2 = pyplot.subplot(2, 1, 2)
pyplot.rcParams['font.sans-serif'] = ['SimHei']
figure2.set_title(u'DBSCAN聚类结果')

type1_distribute = figure2.scatter(type1['Age'], type1['Fare'], c='green', marker='d')
type2_distribute = figure2.scatter(type2['Age'], type2['Fare'], c='red', marker='*')
type3_distribute = figure2.scatter(type3['Age'], type3['Fare'], c='black', marker='p')
type4_distribute = figure2.scatter(type4['Age'], type4['Fare'], c='brown')
pyplot.xlabel(u'某类人的年龄')
pyplot.ylabel(u'某类人的票价')
figure2.legend((type1_distribute, type2_distribute, type3_distribute, type4_distribute), ('第一类', '第二类', '第三类', '第四类'),
               loc=1)

pyplot.show()
