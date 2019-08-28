# sklearn实现机器学习算法记录

需要引入最重要的库：Scikit-learn

## 一、KNN算法

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target

x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=0.3)

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

print(knn.predict(x_test))
print(y_test)
```

## 二、朴素贝叶斯

```python
from sklearn.naive_bayes import BernoulliNB

def loadDataSet():
    '''
    postingList: 进行词条切分后的文档集合
    classVec:类别标签
    使用伯努利模型的贝叶斯分类器只考虑单词出现与否（0，1）
    '''
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1代表侮辱性文字，0代表正常言论
    return postingList, classVec

def create_wordVec(dataset):
    word_set = set([])
    for doc in dataset:
        word_set = word_set | set(doc)  # 通过对两个集合取并，找出所有非重复的单词
    return list(word_set)

def words2Vec(wordList, input_set):
    '''
    @wordList：为前一个函数的输出值（包含单词）
    @input_set：输入需要分类的集合
    函数输出：包含0，1的布尔型向量（对应Wordlist中的单词出现与否）
    '''
    return_vec = [0] * len(wordList)
    # 创建与词汇表等长的列表向量
    for word in input_set:
        if word in wordList:
            return_vec[wordList.index(word)] = 1  # 出现的单词赋1
        else:
            print("the word %s is not in list" % word)
    return return_vec

if __name__ == '__main__':
    p, c = loadDataSet()
    vocab = create_wordVec(p)
    vec = []
    for pl in p:
        vec.append(words2Vec(vocab, pl))

    clf = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)  # 伯努利模型
    clf.fit(vec, c)
    print("预测值：")
    print(clf.predict(vec))
    print("正确值：")
    print(c)
```

## 三、Logistic回归

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

breast_cancer = load_breast_cancer()
# print(diabetes)

diabetes_x = breast_cancer.data
diabetes_y = breast_cancer.target
# print(diabetes_x)
# print(diabetes_y)

x_train, x_test, y_train, y_test = train_test_split(diabetes_x, diabetes_y, test_size=0.3)

log = LogisticRegression(solver='liblinear')
log.fit(x_train, y_train)

print(log.predict(x_test))
print(y_test)
# count = 0
# l = len(y_test)
# print(l)
# for i in range(l):
#     if log.predict(x_test)[i] != y_test[i]:
#         count += 1
# print(count)
#
# print(1 - count / l)  # 输出准确率
```

## 四、支持向量机SVM

### 1. 线性 SVM 分类器

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.svm import SVC

X, y = make_blobs(n_samples=50, centers=2,
                  random_state=0, cluster_std=0.60)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring')

model = SVC(kernel='linear')
model.fit(X, y)

def plot_svc_decision_function(clf, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    if plot_support:
        ax.scatter(clf.support_vectors_[:, 0],
                   clf.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model) #显示分界线

plt.show()

print("分类值：")
print(model.predict(X))

print("正确值：")
print(y)
```

### 2. SVM 与 核函数

对于非线性可切分的数据集，要做分割，就要借助于核函数

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_circles
from sklearn.svm import SVC
from mpl_toolkits import mplot3d

X, y = make_circles(100, factor=0.1, noise=0.1)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring')

# r = np.exp(-(X ** 2).sum(1))
# 画出3D图像
#
# def plot_3D(elev=30, azim=30, X=X, Y=y):
#     ax = plt.subplot(projection='3d')
#     ax.scatter3D(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
#     ax.view_init(elev=elev, azim=azim)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#
#
# plot_3D(elev=45, azim=45, X=X, Y=y)
# plt.show()

model = SVC(kernel='rbf', C=1E6)
model.fit(X, y)


def plot_svc_decision_function(clf, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    if plot_support:
        ax.scatter(clf.support_vectors_[:, 0],
                   clf.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model)

plt.show()

print("分类值：")
print(model.predict(X))

print("正确值：")
print(y)
```

### 3. 总结

1. 非线性映射是SVM方法的理论基础，SVM利用内积核函数代替向高维空间的非线性映射；
2. 对特征空间划分的最优超平面是SVM的目标，最大化分类边际的思想是SVM方法的核心；
3. 支持向量是SVM的训练结果,在SVM分类决策中起决定作用的是支持向量。因此，模型需要存储空间小，算法鲁棒性强；
4. 无任何前提假设，不涉及概率测度；
5. SVM算法对大规模训练样本难以实施；
6. 用SVM解决多分类问题存在困难，经典的支持向量机算法只给出了二类分类的算法，而在数据挖掘的实际应用中，一般要解决多类的分类问题。可以通过多个二类支持向量机的组合来解决。主要有一对多组合模式、一对一组合模式和SVM决策树；再就是通过构造多个分类器的组合来解决。主要原理是克服SVM固有的缺点，结合其他算法的优势，解决多类问题的分类精度。如：与粗集理论结合，形成一种优势互补的多类问题的组合分类器；
7. SVM是O(n^3)的时间复杂度。在sklearn里，LinearSVC是可扩展的(也就是对海量数据也可以支持得不错), 对特别大的数据集SVC就略微有点尴尬了。不过对于特别大的数据集，你倒是可以试试采样一些样本出来，然后用rbf核的SVC来做做分类。