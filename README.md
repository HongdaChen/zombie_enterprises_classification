# zombie_enterprises_classification
using neural_network and logistic_regression model to complete a binary classification

**raw data:**
- 企业基本信息
- 年报数据(三年)
- 融资数据（三年）
- 知识产权数据

**processing data**
首先对我们的原始数据进行检查，发现数据集如下特点：
训练集:僵尸企业:丢失标签=5073:9977 ，接近1:2
![Image text](https://github.com/HongdaChen/zombie_enterprises_classification/blob/master/picture/base.png)
测试集:

Processing_all_data_X对数据进行处理包括：

1.先将后四个有关money的表格合并，是三年的数据，对每个企业ID的三个年份补充全，再将三年的数据分开发现每个年份之间的行数不一样，于是对三年都存在的ID取交集，再对前四个表格的ID取个交集，去掉了大约1000条数据。

2.填充完缺失值后（以众数，平均数），三年取平均再与前四个表格合并，输出为encoded_all_data

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

