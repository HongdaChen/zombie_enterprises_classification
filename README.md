# zombie_enterprises_classification
using neural_network and logistic_regression model to complete a binary classification

### raw data:
- 企业基本信息
- 年报数据(三年)
- 融资数据（三年）
- 知识产权数据

### processing data
在Processing_all_data_X完成

首先对我们的原始数据进行**检查**，发现数据集如下特点:

训练集:僵尸企业:丢失标签=5073:9977 ，接近1:2
![Image text](https://github.com/HongdaChen/zombie_enterprises_classification/blob/master/picture/base.png)

测试集:僵尸企业:正常企业:丢失标签=8928:21650:306,接近2:5
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

-对A组和B组企业ID取交集,然后将两组数据都有的ID部分合并，到此我们所有处理好的数据都合并为一个表，保存至encoded_all_data。


3.Neural_network_X和Logistic_regression_X分别读取encoded_all_datad的有标签数据，输入模型后进行训练，各自再对无flag部分打上标签，最后整体数据的flag情况：十分接近35:65，说明模型贴标签的准确率得到保证

![Image text](https://github.com/HongdaChen/zombie_enterprises_classification/blob/master/pictures/log.png)

![Image text](https://github.com/HongdaChen/zombie_enterprises_classification/blob/master/pictures/neuron.png)

4.利用logistic_regression对全部数据训练，当准确率达到接近98%时，输出权重，作图：

![Image text](https://github.com/HongdaChen/zombie_enterprises_classification/blob/master/pictures/wordcloud.png)

![Image text](https://github.com/HongdaChen/zombie_enterprises_classification/blob/master/pictures/wordcloud2000.png)

![Image text](https://github.com/HongdaChen/zombie_enterprises_classification/blob/master/pictures/weights.png)

4.train_all_data_X包括两个模型的完整训练，模型分别使用对方的完整数据，都分出30%的测试集，准确率如图：

logistic_regression在测试集上的准确率
![Image text](https://github.com/HongdaChen/zombie_enterprises_classification/blob/master/pictures/l_tes_acc.svg)

Neural_network在测试集上的准确率
![Image text](https://github.com/HongdaChen/zombie_enterprises_classification/blob/master/picture/neuron_valid_acc.svg)

