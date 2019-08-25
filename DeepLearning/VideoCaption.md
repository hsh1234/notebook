# 视频描述（Video Caption）近年重要论文总结

## 视频描述

顾名思义视频描述是计算机对视频生成一段描述，如图所示，这张图片选取了一段视频的两帧，针对它的描述是"A man is doing stunts on his bike"，这对在线的视频的检索等有很大帮助。近几年图像描述的发展也让人们思考对视频生成描述，但不同于图像这种静态的空间信息，视频除了空间信息还包括时序信息，同时还有声音信息，这就表示一段视频比图像包含的信息更多，同时要求提取的特征也就更多，这对生成一段准确的描述是重大的挑战。

### 一、long-term Recurrent Convolutional Networks for Visual Recognition and Description --- 2015.2.17

![1566703632542](E:\notebook\DeepLearning\VideoCaption.assets\1566703632542.png)

1. 在本文中提出了Long-term Recurrent Convolutional Network (LRCN)模型，包含了一个特征提取器（例如CNN），以及时序学习器，该模型不是专门用于视频描述的，该文章使用该模型的三种类型用在不同的任务上面。
2. 帧画面通过特征变换参数（特征提取器）得到一个固定长度向量来表示该帧画面的特征，在得到帧画面的特征值后输入到序列模型（例如LSTMs），然后经过softmax进行选词：

![1566713816911](E:\notebook\DeepLearning\VideoCaption.assets\1566713816911.png)

3. 该模型可以适应多种模式：

   ![1566714677161](E:\notebook\DeepLearning\VideoCaption.assets\1566714677161.png)

   1. Sequential inputs, fifixed outputs： many-to-one的模型，实现方式是对于序列模型，在最后步骤合并之前步长所学习到的特征成为一个总的特征y，这样就得到了一个输出。
   2. Fixed inputs, sequential outputs： one-to-many的模型，实现方式是在所有序列模型的输入步长都使用同一个x，由于个步长都会得到一个输出，因此得到了一个序列的输出。
   3. Sequential inputs and outputs： many-to-many的模型，实现方式是采用encoder-decoder的方法，在encoder的时候，每个步骤依次输入不同的x，最终encoder会的到一个固定长度的向量，然后输入到decoder中，产生不固定长度的输出。

4. 训练方法：

   1. 使用随机梯度下降方法对模型进行训练，使输出y落在真实单词位置的可能性最大，也就是最大似然方法。
   2. 采用交叉熵公式：![1566715756310](E:\notebook\DeepLearning\VideoCaption.assets\1566715756310.png)
   3. 使用负对数的方法，变成最小化问题。

5. 指标：

   1. **Activity recognition**

   2. **Image description**

   3. **Video description**：

      在视频描述方面使用了模型如下：

      ![1566719204700](E:\notebook\DeepLearning\VideoCaption.assets\1566719204700.png)

      主要在LSTM之前使用了CRF对视频进行处理，得到如下评估数据（BLEU4）：

      ![1566718674465](E:\notebook\DeepLearning\VideoCaption.assets\1566718674465.png)

      

### 二、Translating Videos to Natural Language Using Deep Recurrent Neural Networks --- 2015.4.30

![1566731313945](E:\notebook\DeepLearning\VideoCaption.assets\1566731313945.png)

#### 模型介绍：

