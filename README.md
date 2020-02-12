# zombie_enterprises_classification
using neural_network and logistic_regression model to complete a binary classification

## raw data:
- 企业基本信息
- 年报数据(三年)
- 融资数据（三年）
- 知识产权数据

## processing data：(在Processing_all_data_X完成)
首先对我们的原始数据进行**检查**，发现数据集如下特点:

**训练集**:僵尸企业:丢失标签=5073:9977 ，接近1:2

![Image text](https://github.com/HongdaChen/zombie_enterprises_classification/blob/master/picture/base.png)

**测试集**:僵尸企业:正常企业:丢失标签=8928:21650:306,接近2:5

![Image text](https://github.com/HongdaChen/zombie_enterprises_classification/blob/master/picture/varify.png)

首先，为了简化数据处理操作，将所有数据分为两组
- 没有year信息的'企业基本信息'和'知识产权数据'合并(命名为A组)
- 包括year信息的'年报数据'和'融资数据'合并（命名为B组）

然后分别处理:

**对B组:**

- 对数据的year缺失值进行填充，包括2015，2016，2017这三年的数据。考虑到不同年份的各种影响因素，所以将三年数据分开进行填充，生成**2015**，**2016**，**2017**。其中，所有者权益合计=资产总额-负债总额，由于'所有者权益合计'缺失值较其他二者少一些，所以最后根据计算公式填充'所有者权益合计'。对于其他特征都采用本年相应特征数据的均值进行填充。
- 完成缺失值的填充,(2015+2016+2017)/3，三表合一。
- 再利用公式$$ /frac{x-mu}{std} $$对各维特征数据标准化。

**对A组:**

- 利用众数对各维数据缺失值填充
- 对文本数据编码, **'行业','企业类型','控制人类型','区域'** 利用sklearn库的LabelEncoder().fit_transform()方法编码
- 考虑到'注册时间'的数据较大，且和编码数据近似，故将2000年定为0年,依次类推。
- '注册资本'的特征利用公式$$ /frac{x-mu}{std} $$标准化

**合并AB**

- 对A组和B组企业ID取交集,然后将两组数据都有的ID部分合并，到此我们所有处理好的数据都合并为一个表，保存至encoded_all_data。

**分离flag为空的数据**

- 有标签数据占全部数据的77%,在精准模型诞生之前，如果采用某种手段将其余23%的flag补充上，势必会带来误差，即使误差很小，也改变了实际情况，以此为基础的模型学习的也不再是实际情况。77%（35648条）完整数据可用,基本满足后续模型需求。

## features engineering

- 现在我们的这个数据表格有23个特征列,数据维度比较高,对于neural_network模型来说是可接受的，但为了让我们的数据适应更多的模型,同时又具有很好的可解释性，需要对主要特征进行提取。以下统计图将各个特征与flag的情况可视化出来，橙色代表僵尸企业，蓝色代表正常企业。由图可见，**纳税总额**和**净利润**对分类作用最为明显，其次是**主营业务收入，内部融资和贸易融资额度，股权融资成本，股权融资额度**。而其他特征作用不明显，可以考虑舍弃，经过后面模型的验证，这样做确实是合理的。

![Image text](https://github.com/HongdaChen/zombie_enterprises_classification/blob/master/picture/下载.png)

- 提取出这些特征，再次做统计图，可以更清晰看出各个特征对分类的贡献。

![Image text](https://github.com/HongdaChen/zombie_enterprises_classification/blob/master/picture/下载(1).png)

- 提取出的这些特征

![Image text](https://github.com/HongdaChen/zombie_enterprises_classification/blob/master/picture/quan.png)


## classification_model

- ***logistic regression***

logistic回归是一种广义线性回归（generalized linear model），与多重线性回归分析有很多相同之处。它们的模型形式基本上相同，都具有 wx + b，其中w和b是待求参数，其区别在于他们的因变量不同，多重线性回归直接将wx+b作为因变量，即y =wx+b,而logistic回归则通过函数L将wx+b对应一个隐状态p，p =L(wx+b),然后根据p 与1-p的大小决定因变量的值。如果L是logistic函数，就是logistic回归。简单来说，就是logistic回归会在线性回归后再加一层logistic函数的调用。logistic回归主要是进行二分类预测，Sigmod函数是最常见的logistic函数，因为Sigmod函数的输出的是是对于0~1之间的概率值，当概率大于0.5预测为1，小于0.5预测为0。

,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,明天再写,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

模型训练结果:

![Image text](https://github.com/HongdaChen/zombie_enterprises_classification/blob/master/picture/logistestacc.png)

![Image text](https://github.com/HongdaChen/zombie_enterprises_classification/blob/master/picture/logtrianloss.png)


- ***neural network*** 

,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,明天再写,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

模型训练结果:

![Image text](https://github.com/HongdaChen/zombie_enterprises_classification/blob/master/picture/neuronvalidacc.png)

![Image text](https://github.com/HongdaChen/zombie_enterprises_classification/blob/master/picture/neutrainloss.png)


- ***SVM***

,,,,,,

- ***K-means***

,,,,,,


## Enterprise image

![Image text](https://github.com/HongdaChen/zombie_enterprises_classification/blob/master/picture/leida.png)


## Find hyper-parameters about optimization

,,,,,,,



