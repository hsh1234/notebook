# 视频描述（Video Caption）近年重要论文总结

## 视频描述

顾名思义视频描述是计算机对视频生成一段描述，如图所示，这张图片选取了一段视频的两帧，针对它的描述是"A man is doing stunts on his bike"，这对在线的视频的检索等有很大帮助。近几年图像描述的发展也让人们思考对视频生成描述，但不同于图像这种静态的空间信息，视频除了空间信息还包括时序信息，同时还有声音信息，这就表示一段视频比图像包含的信息更多，同时要求提取的特征也就更多，这对生成一段准确的描述是重大的挑战。

### 一、long-term Recurrent Convolutional Networks for Visual Recognition and Description --- 2015.2.17

![1566703632542](.\VideoCaption.assets\1566703632542.png)

1. 在本文中提出了Long-term Recurrent Convolutional Network (LRCN)模型，包含了一个特征提取器（例如CNN），以及时序学习器，该模型不是专门用于视频描述的，该文章使用该模型的三种类型用在不同的任务上面。
2. 帧画面通过特征变换参数（特征提取器）得到一个固定长度向量来表示该帧画面的特征，在得到帧画面的特征值后输入到序列模型（例如LSTMs），然后经过softmax进行选词：

![1566713816911](.\VideoCaption.assets\1566713816911.png)

3. 该模型可以适应多种模式：

   ![1566714677161](.\VideoCaption.assets\1566714677161.png)

   1. Sequential inputs, fifixed outputs： many-to-one的模型，实现方式是对于序列模型，在最后步骤合并之前步长所学习到的特征成为一个总的特征y，这样就得到了一个输出。
   2. Fixed inputs, sequential outputs： one-to-many的模型，实现方式是在所有序列模型的输入步长都使用同一个x，由于个步长都会得到一个输出，因此得到了一个序列的输出。
   3. Sequential inputs and outputs： many-to-many的模型，实现方式是采用encoder-decoder的方法，在encoder的时候，每个步骤依次输入不同的x，最终encoder会的到一个固定长度的向量，然后输入到decoder中，产生不固定长度的输出。

4. 训练方法：

   1. 使用随机梯度下降方法对模型进行训练，使输出y落在真实单词位置的可能性最大，也就是最大似然方法。
   2. 采用交叉熵公式：![1566715756310](.\VideoCaption.assets\1566715756310.png)
   3. 使用负对数的方法，变成最小化问题。

5. 指标：

   1. **Activity recognition**

   2. **Image description**

   3. **Video description**：

      在视频描述方面使用了模型如下：

      ![1566719204700](.\VideoCaption.assets\1566719204700.png)

      主要在LSTM之前使用了CRF对视频进行处理，得到如下评估数据（BLEU4）：

      ![1566718674465](.\VideoCaption.assets\1566718674465.png)

      

### 二、Translating Videos to Natural Language Using Deep Recurrent Neural Networks --- 2015.4.30

![1566731313945](.\VideoCaption.assets\1566731313945.png)

#### 模型介绍：

先对所有视频帧画面使用卷积神经网络进行图片特征提取，获取fc7层的特征向量（4096固定长度），然后将所有帧画面提取到的特征向量做meanpooling得到一个最终向量（类似图片描述中的输入向量）。在LSTMs网络中，每个步长都输入同样的向量，并在每个步长都得到LSTMs的一个输出作为当前输出单词的编码，直到输出结束符<EOS>为止。

在本文中提到，在Donahue et al. (2014)提出两层的LSTM比四层或者单层的LSTM效果好。

对于单词的处理方式：one-hot编码

#### 训练方法：

采用最大似然法，优化参数的值，来最大化生成正确句子的概率。 given the corresponding video *V* and the model parameters *θ*，对应的交叉熵公式：![1566784464077](E:\notebook\DeepLearning\VideoCaption.assets\1566784464077.png)

上式是对于整个句子做交叉熵，在本文中还可以对每个单词做交叉熵后相加得到损失值：

![1566784597082](.\VideoCaption.assets\1566784597082.png)

对于将LSTM的输出映射到one-hot词库还是使用softmax函数：

![1566789153944](.\VideoCaption.assets\1566789153944.png)

#### 评估指标：

数据集使用：MSVD

指标：

![1566789433801](.\VideoCaption.assets\1566789433801.png)

在本文中去掉了mean pooling，直接输入单个帧特征到模型中，查看mean pooling的影响，最终效果：

![1566789877545](.\VideoCaption.assets\1566789877545.png)

相对没有mean pooling差。

#### 总结：

缺点：

1. 使用mean pooling对所有视频帧整合，丢失了视频序列上的部分信息。



### 三、Sequence to Sequence – Video to Text --- 2015.10.19

![1566790266376](.\VideoCaption.assets\1566790266376.png)

![1566790279497](.\VideoCaption.assets\1566790279497.png)

#### 模型介绍：

本文是早期经典文章，思路相对简单，如图所示，对视频的特征提取也仅仅对每帧的图像使用CNN网络进行2D特征的提取，同时加入了另外的特征——光流图像提取的特征，因为可以更好的表示视频中的动作，整个视频encoder和decoder过程在一个LSTM模型上完成，考虑到了视频的时序特征，因此使用LSTM网络来顺序进行图像特征的输入，用隐含层来表示整个视频，再接着输入单词来逐个预测单词，之后是详细介绍。

本文提出的模型，对于视频抽取帧画面之后，使用训练好的VGG16模型对帧画面进行特征提取，得到fc7层的输出向量（4096长度），然后按视频帧顺序依次输入到LSTMs网络中，在输入过程中不产生输出，做encoding操作，并将第一层得到的输出向量输入到第二层LSTM，当所有视频帧输入完毕，开始获取第二层LSTM的输出（也就是对应的描述句子单词），直到获得<eos>为止。

#### 训练方法：

采用最大似然法，优化参数的值，来最大化生成正确句子的概率，对于第二层LSTM的输出，经过softmax到one-hot词库中取词，并通过交叉熵的方式来计算误差值：![1566798383134](.\VideoCaption.assets\1566798383134.png)

![1566798417750](.\VideoCaption.assets\1566798417750.png)

#### 评估指标：

使用视频数据集：MSVD

![1566799051366](.\VideoCaption.assets\1566799051366.png)

#### 总结：

由于是早期的文章，忽略了很多东西，比如在image caption中有显著贡献的attention机制，更好的时序特征提取技术，其他的特征比如语音、背景音等特征。可以说这篇文章极大的依赖LSTM网络本身的性质，时序特征也就是image feature之间的关联也靠模型自动学习，包括最终的视频特征和之后单词之间的关联也都靠LSTM模型自动学习，作者只加了一个光流图像特征进行加权平均。



### 四、**Video Paragraph Captioning Using Hierarchical Recurrent Neural Networks** --- 2016.4.6

![1566981985654](.\VideoCaption.assets\1566981985654.png)

#### 模型介绍：
