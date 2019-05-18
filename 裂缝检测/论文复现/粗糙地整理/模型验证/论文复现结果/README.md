[TOC]

# 结果说明：

实验训练的训练数据集为 ：正样本40w+，负样本50w+

实验生成的网络模型较小，为377KB

实验召回率和精确度分别为：`PRE: 0.9373 REC: 0.9331`,详情可查看 `Training_procedure.txt`

实验结果中(见ResultImage)，黑色为裂缝，白色为非裂缝， 其中，边框的黑色是为了使用图片的所有信息而添加的填充，从图片中可看出，实验结果明显地表示了裂缝的纹理，但也显示了很多的noise，如原始图片的阴影，油印。

![](.\ResultImage\032.jpg)

# 需要改进的地方

论文中的裂缝纹理更加细致，且受noise的影响更低，因此需要继续优化

![论文图片](.\paperImage\论文图片.png)



# 实验的可应用性

实验生成图片的过程是：对以当前像素为**中心**，采样27\*27的图片，将得到的结果拼成一张完整的图片，原始的图片大小为320*480，生成一个像素点需要一个27\*27的子图片，即生成一张结果图需要对1.5w张图片进行分类，输出结果。虽然网络模型较小，但也需要3分钟/张。（应该还可以继续优化，但时间消耗应该不低

- 尝试过的检测方法：直接输入一张图片，将图片切块进行分类，但分类效果很差,可以算是基本没有效果

  ![](.\paperImage\10.jpg)

  原因分析：如下图的红绿框，裂缝经过了红色框的中心，没有经过绿色框的中心，所以红色框中心像素标记为“裂缝”（黑色），绿色框中心像素标记为“非裂缝”（白色），因此，裂缝需要经过分类框的中心才能检测为裂缝，不经过中心，即使包含裂缝，也分类为非裂缝

  ![在这里插入图片描述](https://img-blog.csdnimg.cn/20190518161340714.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MzAzODYy,size_16,color_FFFFFF,t_70)

- 未尝试的方法1：根据检测框的阈值来判断是否为裂缝，像素，裂缝像素为0，非裂缝像素为1，即通过矩阵求和，小于一定的值，可以判定为裂缝

- 未尝试的方法2：图片与图像分割有点类似



# REFERENCES

[Automatic  Pavement Crack Detection Based on Structured Prediction with the Convolutional Neural Network    2018  CVPR](https://arxiv.org/abs/1802.02208)

