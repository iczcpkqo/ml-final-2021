# Final Assignment 2021

> Student Name: Xiang Mao
> Student Number: 21332237

## Q1

### 选择的数据集是什么?

我选择的位于城市中心的车站是位于所有车站纬度中位数附近的车站, 市郊的车站为所有车站中经度最小的车站. 如下:

1. Station: Exchequer Street
   * Number: 9
   * Latitude: 53.343033
   * Longitude: -6.263578
   * Location: In the city centre

2. Station: Kilmainham Gaol
   * Number: 97
   * Latitude: 53.342113
   * Longitude: -6.310015
   * Location: In the suburbs

我选择这两个车站的原因是:

1. 两个车站的地理位置需要足够的远, 从而有差异尽可能大的行为数据;
2. 位于市郊的车站位于经度最小的地方, 从而确保距离市中心足够远;
3. 位于市中心的车站的坐标在所有车站纬度的中位数附近, 从而让这个车站尽可能体现出市中心车站的行为数据.

### 特征工程

* 原始数据共有11个特征，这些特征有如下关系：
  1. 'Name'，'ADDRESS'与'STATION ID'一一对应;
  2. $BIKE STANDS = AVAILABLE BIKE STANDS + AVAILABLE BIKES$
  3. 对于'STATUS'，其中 `STATUS=open` 的数据有数据。
  4. 对同一个车站，'LATITUDE' 及'LONGITUDE' 的取值都是固定的，因此，该特征是不影响预测占用率的无效特征, 所以在预测时删除这两个特征.

* 构造新特征, 如下:
$$
STAND\_OCCUPANCY = \frac{(BIKE STANDS-AVAILABLE BIKES)}{BIKE STANDS}
$$

* 综上所述, 使用的特征为: 'STATION ID', 'TIME', 'BIKE STANDS', 'AVAILABLE BIKE', 'STANDS', 'STAND_OCCUPANCY'. 

* 使用的labels如下:
  1. 'STAND_OCCUPANCY_10': 在对应时间的10分钟后的占用率.
  2. 'STAND_OCCUPANCY_30': 在对应时间的30分钟后的占用率.
  3. 'STAND_OCCUPANCY_60': 在对应时间的60分钟后的占用率.

* 下图为 'STATION ID', 'AVAILABLE BIKE STANDS'以及 'STAND_OCCUPANCY' 与 label('STAND_OCCUPANCY_10')之间的热力图:

    ==图==

由相关系数的大小可以发现，'STAND_OCCUPANCY'与lable高度相关，是十分重要的特征, 'STAND_OCCUPANCY'与'AVAILABLE BIKE STANDS'的相关系数为0.85，也有较高的相关性。



### LSTM方法

我是用的两种机器学习方法分别为: LSTM和随机森林.

LSTM(Long short-term memory) 是一种特殊的RNN，使用这种方法可以用于解决长序列训练过程中出现的梯度爆炸问题和梯度消失问题. LSTM相比普通的RNN, 能够在更长的序列中获得更好的结果.

loss function使用MSE，优化器如下:

```python
  optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
```

上面的优化器使用了`Adam(Adaptive Moment Estimation)`算法. Adam(Adaptive Moment Estimation)本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。它的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。

* 第一个参数为: 可用于迭代优化的参数或者定义参数组的dicts.
* 第二个参数lr为: 学习率(默认: 0.001)

* LSTM参数
  1. input_size: 对应的及特征数量.
  2. output_size: 预测变量的个数和数据标签的个数.
  3. hidden_layer_size: 隐藏层的特征数，也就是隐藏层的神经元个数.
  4. BATCH_SIZE: 这个参数的大小将决定进行一次训练的样本数目. 它将影响到模型的优化速度和优化程度。

```python
  nn.LSTM(input_size,hidden_layer_size)
  nn.Linear(hidden_layer_size,output_size)
```

#### 97站

基线为一个每次都预测占用率平均值的模型.

##### 10分钟预测随机对比

  ==10分钟预测随机对比==

Figure 1.2.1 为'10分钟预测结果的随机对比'

通过上图的结果可以看到预测未来10分钟的效果, 预测准确率较高, 因此使用LSTM方法对未来10分钟占用率进行预测是可行的.

### 随机森林方法

使用随机森林进行预测, 使用的MSE作为评估指标, 以下是随机森林中选取的参数:

```python
  bootstrap=True, criterion='mse', max_depth=None,
  max_features='auto', max_leaf_nodes=None,
  min_impurity_decrease=0.0,
  min_impurity_split=None,
  min_samples_leaf=1, min_samples_split=4,
  min_weight_fraction_leaf=0.0,
  n_estimators=200, n_jobs=1,
  oob_score=False, random_state=None,
  verbose=0, warm_start=False
```

  1. bootstrap: 在创建树的过程中, 是否使用自举样本抽样的方式.
  2. criterion: 用于衡量分枝质量的指标.
  3. max_depth: 树的最大深度. 如果是`None`, 树会持续生长知道所有叶子节点的不纯度为0, 或者直到每个叶子节点所含的样本量都小于参数`min_samples_split`中输入的内容.
  4. min_samples_split: 一个中间节点要产生分枝所需要的最小样本量, 如果一个节点包含的样本量小于min_samples_split中填写的数字, 这个节点就不会产生分枝.

#### 97站台

##### 10分钟 随机森林参数

#### 10, 30 ,60, 随机对比

#### 10, 30 ,60, 连续对比

Figure 1.3.1: 10分钟 随机森林参数
Figure 1.3.2: 10, 30 ,60, 随机对比
Figure 1.3.3: 10, 30 ,60, 连续对比

==97随机森林图==

上图为`Station: Kilmainham Gaol`使用随机森林时对未来10分钟,30分钟和60分钟的预测效果. 通过上图的结果和基线的对比可以知道, 预测结果明显好于基线结果, 因此使用随机森林方法对未来10分钟, 30分钟和60分钟的预测是可行的.

#### 9站台

#### 10, 30 ,60, 随机对比

#### 10, 30 ,60, 连续对比

#### 不同站台随机森林效果对比

==9随机森林图==

上图为`Station: Exchequer Street`使用随机森林时对未来10分钟,30分钟和60分钟的预测效果. 通过上图的结果和基线的对比可以知道, 预测结果明显好于基线结果, 因此使用随机森林方法对未来10分钟, 30分钟和60分钟的预测是可行的.

* 通过LSTM和随机森林预测具备不同行为数据的车站的未来10分钟, 30分钟和60分钟的预测效果明显好于每次都预测平均值的基线, 因此通过LSTM和随机森林进行预测是可行的.

## Q2

### What is a ROC curve. How can it be used to evaluate the performance of a classifier.
> 什么是ROC曲线。如何用它来评估一个分类器的性能。

1. ROC是Receiver Operating Characteristic的缩写. ROC的形式是一个画在Cartesian coordinate system中的曲线, 这个曲线叫ROC曲线. 横坐标是false positive rate(FPR)，纵坐标是true positive rate(TPR).
$$
    TPR = TP/(TP+FN) \\
    FPR = FP/(FP+TN)
$$
TP表示预测为真,实际也是真; TN表示预测为假, 实际是假; FP表示预测为真, 实际是假; FN为预测为假, 实际是真.
可以使用AUC（Area Under roc Curve）用来衡量ROC曲线的好坏. AUC的值表示, ROC曲线下方于横轴的面积大小.

### Give two examples of situations when a linear regression would give inaccurate predictions. Explain your reasoning.
> 给出两个例子，说明线性回归会给出不准确的预测的情况。解释你的理由。

当数据分布不符合线性回归的假设条件时，就会导致预测结果不准确的问题。

 1. 线性
当使用线性回归模型时，希望数据中的预测变量x和响应变量y存在线性的关系。然后在当x只有一维的情况时通过一条直线, 或者当x是高维的情况时通过一个面, 用于解释数据才比较合理。 在下两个图中，图1明显存在线性关系，图2则不存在线性关系。所以1图有可能可以使用线性回归，而B图不应该直接使用线性回归。

![image-20220104233135580](D:/sync/dropbox/im/notes/data/resource/image/image-20220104233135580.png)

![image-20220104233112405](D:/sync/dropbox/im/notes/data/resource/image/image-20220104233112405.png)

2. 方差

假设我们想了解家庭收入是如何影响豪车消费的, 其中家庭收入是预测变量, 豪车消费额是响应变量.
对于大多数贫穷的家庭而言,  购买豪车是困难的, 所以对于贫穷家庭而言, 豪车的消费额是很小的, 所以其方差也很小.
对于特别富有的家庭. 购买豪车是没有困难的, 因此是否购买豪车仅仅和不同家庭中的个人喜好有关系了, 因此豪车消费额的方差会很大.

线性回归的结果是由每一个变量和其权重共同决定的, 权重越大, 则对结果的影响越大. 所以结果基本上都被少数几个权重很高的变量决定了. 这样建立出来的模型不仅损失了原始数据中的信息.

### Discuss three pros/cons of an SVM classiﬁer vs a neural net classiﬁer.
> 讨论SVM分类器与神经网络分类器的三个优点/缺点。

1. 神经网络

 * 优点：
    a) 学习规则简单，便于计算机实现。
    b) 神经网络有很强的非线性拟合能力，可映射任意复杂的非线性关系，
    c) 具有很强的记忆能力、强大的自学习能力和非线性映射能力，所以应用面很广.

  * 缺点：
    a) 难以解释推理过程和推理依据
    b) 当数据不够的时候，神经网络就无法进行工作。
    c) 由于所有的问题和特征都转换成了数字, 所以所有的推理都是数值计算, 因此会丢失一些信息.

2. SVM的优缺点

  * 优点：
    a) SVM的训练结果是支持向量, 支持向量在SVM分类决策中起了决定性的作用.
    b) SVM的最终决策函数是由少数的支持向量所确定的, 计算的复杂性并不取决于样本空间的维数, 而是支持向量的数目, 这样可以尽量避免“维数灾难”.
    c) 由于SVM方法的理论基础是非线性映射, SVM利用内积核函数代替向高维空间的非线性映射.

  * 缺点：
    a) 对于大规模训练样本而言, 使用SVM将耗费大量的运算时间和机器内存.
    b) 由于经典的SVM算法值有二分类的算法, 所以在解决多分类问题时, SVM会很麻烦和困难.


### Describe the operation of a convolutional layer in a convNet. Give a small example to illustrate.
> 描述 convNet 中卷积层的操作。举一个小例子来说明。

如图(Figure 2.1)所示，现在我们有一个 $4\times4$ 的图片，使用一个$3\times3$的卷积核对其进行卷积操作，移动步长为1。

1. 卷积核与图片(Figure 2.1)的左上角$3\times3$ 的子矩阵（阴影部分）做点乘，得到结果矩阵中（1，1）位置的值。
2. 卷积核向右滑动一格后，继续与该子矩阵做点乘，得到结果矩阵（1，2）位置的值, 如图(Figure 2.2)。

上述两步将行遍历完成，下移一步，继续遍历行。
得到图(Figure 2.3)，(Figure 2.4)中的结果。
最终得到结果矩阵。

![image-20220105002949416](D:/sync/dropbox/im/notes/data/resource/image/image-20220105002949416.png)

![image-20220105003125368](D:/sync/dropbox/im/notes/data/resource/image/image-20220105003125368.png)

![image-20220105003207457](D:/sync/dropbox/im/notes/data/resource/image/image-20220105003207457.png)

![image-20220105003224197](D:/sync/dropbox/im/notes/data/resource/image/image-20220105003224197.png)

### In k-fold cross-validation a dataset is resampled multiple times. What is the idea behind this resampling i.e. why does resampling allow us to evaluate the generalisation performance of a machine learning model. Give a small example to illustrate.
> 在k-fold交叉验证法中，数据集被多次重采样。这种重采样背后的想法是什么，也就是说，为什么重采样能让我们评估机器学习模型的泛化性能。请举一个小例子来说明。







