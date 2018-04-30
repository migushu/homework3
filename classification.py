from function import *

##########
# 分类预测
##########
TrainSet = pandas.read_csv(open('test.csv'))
TestSet = pandas.read_csv(open('train.csv'))

RedundantAttribute = ['PassengerId', 'Name']
NominalAttribute = ['Pclass', 'Sex', 'Cabin', 'Embarked']
NumericAttribute = ['SibSp', 'Age', 'Parch', 'Fare']

# 丢弃有空值行
TrainSet = TrainSet.dropna(axis=0, how='any')

# Sex属性male转为1，female转为0
TrainSet.replace(to_replace='male', value=1, inplace=True)
TrainSet.replace(to_replace='female', value=0, inplace=True)
TestSet.replace(to_replace='male', value=1, inplace=True)
TestSet.replace(to_replace='female', value=0, inplace=True)

# Age属性nan填充均值
TrainSet['Age'] = TrainSet['Age'].fillna(value=TrainSet['Age'].mean())
TestSet['Age'] = TestSet['Age'].fillna(value=TestSet['Age'].mean())

# Fare属性nan填充均值
TestSet['Fare'] = TestSet['Fare'].fillna(value=TestSet['Fare'].mean())

# 把字符型标称属性映射为数值型
to_encode_attr = ['Embarked', 'Cabin', 'Ticket'];
TrainSet = encode_target(TrainSet, to_encode_attr)
TestSet = encode_target(TestSet, to_encode_attr)

# 把标签修改成true false
# TrainSet['Survived'] = TrainSet['Survived'].replace(to_replace=1, value=True)
# TrainSet['Survived'] = TrainSet['Survived'].replace(to_replace=0, value=False)

# 把Age与Fare修改为整数
TrainSet[['Age', 'Fare']] = TrainSet[['Age', 'Fare']].astype(int)
TestSet[['Age', 'Fare']] = TestSet[['Age', 'Fare']].astype(int)

# 丢弃多余属性列
TrainSet = TrainSet.drop(columns=RedundantAttribute)
TestSet = TestSet.drop(columns=RedundantAttribute)

# 特征属性与标签属性
LabelAttr = ['Survived']
FeatureAttr = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked', 'Ticket'];
Label = TrainSet[LabelAttr]
Feature = TrainSet[FeatureAttr]

# 定义决策树并训练
# DesicionTree = DecisionTreeClassifier(min_samples_split=20, random_state=99)
DesicionTree = DecisionTreeClassifier(criterion='entropy', max_depth=10)
DesicionTree.fit(Feature, Label)

# 使用决策树分类
result1 = DesicionTree.predict(TestSet)
# print(result1)

# 定义高斯分布的朴素贝叶斯分类器
GaussianNaiveBayesClassifier = GaussianNB().fit(Feature, Label)

# 使用高斯朴素贝叶斯分类器分类
result2 = GaussianNaiveBayesClassifier.predict(TestSet)
# print(result2)

# 比较两种分类器的分类结果
if len(result1) == len(result2):
    print('总人数:', Series(result2).count(), '决策树判断生还者:', Series(result1).sum(), '贝叶斯判断生还者:', Series(result2).sum(),
          '两者的相同预测数量:',
          (Series(result1) == Series(result2)).sum())

# 显示分类的结果，横坐标为年龄、纵坐标为票价、红点为死亡、绿点为存活
# 先显示决策树的预测结果
pyplot.figure()

temp = pandas.concat([DataFrame(result1, columns=['Survived']), TestSet[['Age', 'Fare']]], axis=1)
alive = temp.loc[temp['Survived'] == 1]
dead = temp.loc[temp['Survived'] == 0]

figure1 = pyplot.subplot(2, 1, 1)
pyplot.rcParams['font.sans-serif'] = ['SimHei']
figure1.set_title(u'决策树预测结果')
alive_distribute = figure1.scatter(alive['Age'], alive['Fare'], c='green', marker='d')
dead_distribute = figure1.scatter(dead['Age'], dead['Fare'], c='red', marker='*')
pyplot.xlabel(u'生还者或遇难者年龄')
pyplot.ylabel(u'生还者或遇难者票价')
figure1.legend((alive_distribute, dead_distribute), ('生还者', '遇难者'), loc=1)

temp = pandas.concat([DataFrame(result2, columns=['Survived']), TestSet[['Age', 'Fare']]], axis=1)
alive = temp.loc[temp['Survived'] == 1]
dead = temp.loc[temp['Survived'] == 0]

figure2 = pyplot.subplot(2, 1, 2)
pyplot.rcParams['font.sans-serif'] = ['SimHei']
figure2.set_title(u'高斯贝叶斯预测结果')
alive_distribute = figure2.scatter(alive['Age'], alive['Fare'], c='green', marker='d')
dead_distribute = figure2.scatter(dead['Age'], dead['Fare'], c='red', marker='*')
pyplot.xlabel(u'生还者或遇难者年龄')
pyplot.ylabel(u'生还者或遇难者票价')
figure2.legend((alive_distribute, dead_distribute), ('生还者', '遇难者'), loc=1)

pyplot.show()
