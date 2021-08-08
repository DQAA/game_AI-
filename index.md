# 一、比赛背景

第三届中国AI+创新创业大赛由中国人工智能学会主办，半监督学习目标定位竞赛分赛道要求选手基于少量有标注数据训练模型，使分类网络具有目标定位能力，实现半监督目标定位任务。

中国人工智能学会（Chinese Association for Artificial Intelligence，CAAI）成立于1981年，是经国家民政部正式注册的我国智能科学技术领域唯一的国家级学会，是全国性4A级社会组织，挂靠单位为北京邮电大学；是中国科学技术协会的正式团体会员，具有推荐“两院院士”的资格。

中国人工智能学会目前拥有51个分支机构，包括43个专业委员会和8个工作委员会，覆盖了智能科学与技术领域。学会活动的学术领域是智能科学技术，活动地域是中华人民共和国全境，基本任务是团结全国智能科学技术工作者和积极分子通过学术研究、国内外学术交流、科学普及、学术教育、科技会展、学术出版、人才推荐、学术评价、学术咨询、技术评审与奖励等活动促进我国智能科学技术的发展，为国家的经济发展、社会进步、文明提升、安全保障提供智能化的科学技术服务。

中国“AI+”创新创业大赛由中国人工智能学会发起主办，是为了配合实施创新驱动助力工程，深入开展服务企业技术创新活动，进一步提高我国文化建设和实践创新能力，展示智能科学与技术等相关学科建设的新经验、新成果，促进专业内涵的建设而发起的综合性大赛平台。

飞桨PaddlePaddle作为中国首个自主研发、功能完备、开源开放的产业级深度学习平台，为本次比赛的参赛选手提供了集深度学习核心训练和推理框架、基础模型库、端到端开发套件和丰富的工具组件于一体的一站式服务。百度大脑AI Studio作为官方指定且唯一的竞赛日常训练平台，为参赛选手提供高效的学习和开发环境，更有亿元Tesla V100算力免费赠送，助力选手取得优异成绩。

[比赛链接](https://aistudio.baidu.com/aistudio/competition/detail/78)

# 二、简要介绍
## 1.赛题背景
半监督学习(Semi-Supervised Learning)是指通过大量无标记数据和少量有标记数据完成模型训练，解决具有挑战性的模式识别任务。近几年，随着计算硬件性能的提升和大量大规模标注数据集的开源，基于深度卷积神经网络(Deep Convolutional Neural Networks, DCNNs)的监督学习研究取得了革命性进步。然而，监督学习模型的优异性能要以大量标注数据作为支撑，可现实中获得数量可观的标注数据十分耗费人力物力(例如，获取像素级标注数据)。于是， 半监督学习逐渐成为深度学习领域的热门研究方向，只需要少量标记数据就可以完成模型训练过程，更适用于现实场景中的各种任务。
## 2.比赛任务
本次比赛要求选手基于少量有标签的数据训练模型，使分类网络具有目标定位能力，实现半监督目标定位任务。每- -位参赛选手仅可以使用ImageNet大型视觉识别竞赛(LSVRC)的训练集图像作为训练数据，其中有标签的训练数据仅可以使用大赛组委会提供的像素级标注。
## 3.数据集介绍
* 训练数据集包括50,000幅像素级标注的图像，共包含500个类，每个类100幅图像;
* A榜测试数据集包括1 1,878幅无标注的图像;
* B榜测试数据集包括10,989幅无标注的图像。
## 4.评价指标
本次比赛使用loU曲线作为评价指标，即利用预测的目标的定位概率图，计算不同阈值下预测结果与真实目标之间的IoU分数，最后取一个最高点作为最终的分数。在理想状态下，loU曲线最高值接近1.0,对应的阈值为255,因为阈值越高，目标对象与背景的对比度越高。

# 三、代码内容说明
见notebook

# 四、模型构建思路及调优过程
## 1.完整算法结构框图、思路步骤详述、代码组织结构介绍

DeepLabv3+是DeepLab系列的最后一篇文章，其前作有DeepLabv1、DeepLabv2和DeepLabv3。在最新作中，作者结合编码器-解码器(encoder-decoder)结构和空间金字塔池化模块(Spatial Pyramid Pooling, SPP)的优点提出新的语义分割网络DeepLabv3+，在 PASCAL VOC 2012和Cityscapes数据集上取得新的state-of-art performance. 其整体结构如下所示，Encoder的主体是带有空洞卷积(Atrous Convolution)的骨干网络，骨干网络可采用ResNet等常用的分类网络，作者使用了改进的Xception模型作为骨干网络。紧跟其后的空洞空间金字塔池化模块(Atrous Spatial Pyramid Pooling, ASPP)则引入了多尺度信息。相比前作DeepLabv3，DeepLabv3+加入decoder模块，将浅层特征和深层特征进一步融合，优化分割效果，尤其是目标边缘的效果。此外，作者将深度可分离卷积(Depthwise Separable Convolution)应用到ASPP和Decoder模块，提高了语义分割的健壮性和运行速率。  
![DeepLabv3+结构图](https://ai-studio-static-online.cdn.bcebos.com/dafd9c0f06ab41e8af8421c53a0778841fe43ba408f04382885db668db8d00a9)  

FCN（Fully Convolutional Network for Semantic Segmentation）可以对图像进行像素级的分类，解决了语义级别的图像分割问题，因此现有的大多数语义分割方法都基于FCN。但这些方法也有一定缺陷，比如分辨率低、上下文信息缺失和边界错误等。2020年，相关学者为解决语义分割上下文信息缺失难题，建设性地提出OCRNet，即基于物体上下文特征表示（Object Contextual Representation，以下简称OCR）的网络框架。其整体结构如下所示。实现此OCR方法需要经历三个阶段——首先形成软物体区域（Soft Object Regions），然后计算物体区域表示（Object Region Representations），最后得到物体上下文特征表示和上下文信息增强的特征表示（Augmented Representation）。 与其他语义分割方法相比，OCR方法更加高效准确。因为OCR方法解决的是物体区域分类问题，而非像素分类问题，即OCR方法可以有效地、显式地增强物体信息。从性能和复杂度来说，OCRNet也更为优秀。2020年，“HRNet + OCR + SegFix”版本在2020ECCV Cityscapes 获得了第一名。  
![OCRNet结构图](https://ai-studio-static-online.cdn.bcebos.com/dc2e6f01a151449d89f5f520f90f43a9009d7373c3ad4e71831a9d02204e70ae)  

完整算法结构框图如下图所示。  
![完整算法结构框图](https://ai-studio-static-online.cdn.bcebos.com/4a937eafce33447eb3735e81c783feb50571f2a2e81744359359619136bc2ce6)

	本方案在PaddleSeg的基础上使用了三个OCRNet+HRNet_W48，一个Deeplabv3p+ResNet101_vd共四个模型进行融合。4个模型各自对输入图像进行预测得到4个结果，然后进行投票，即对4个结果取平均作为最终结果，最终结果中像素值越大，表示有越多模型认为该像素为前景。该方案在B榜的得分为0.77732。
    
    比赛初期，我们使用了Deeplabv3+ResNet50的结构，最终在A榜的得分为0.75+。后来尝试了Deeplabv3+ResNet101_vd，Deeplabv3p+ResNet50，Deeplabv3p+ResNet101_vd等结构，最终A榜得分0.76+。在这基础上，又使用了Deeplabv3p+ResNet101_vd的结构，并且将图像的Resize大小由[256,256]改为[512,512]，并对学习率和损失函数进行了调整，最终A榜得分0.77423，B榜得分0.76455。后来进一步使用了OCRNet+HRNet_W48，B榜得分0.7678。最后使用3个OCRNet+HRNet_W48和1个Deeplabv3p+ResNet101_vd融合B榜得分0.77732。
    
    代码存放在/home/aistudio/work/下。PaddleSeg/为PaddleSeg图像分割库，其中部分代码进行了修改以便于完成比赛任务，configs/为4个模型的配置文件，log/为训练一个epoch的日志及4个模型评估的完整日志，data_path/为训练数据、验证数据、测试数据的路径文件，model/为4个模型的权重。数据存放于/home/aistudio/data/下。
    
## 2.数据增强及清洗策略
4个模型除Resize的大小不同外均使用了相同的数据增强策略。  
1. 随机水平翻转
2. 随机垂直翻转
3. 随机失真：亮度、对比度、饱和度
4. 标准化
5. Resize: 3个OCRNet的Resize大小为[256,256]，Deeplabv3p的Resize大小为[512,512]  

对于数据划分方面，OCRNet1和Deeplabv3p使用40000张图片训练，10000张图片评估，OCRNet2和OCRNet3使用45000张图片训练，5000张图片评估。对于label图片，我们将图片转成单通道的灰度图，并将像素值改为0和1。划分数据集时，我们等比例地在每个类中进行划分，以保证各个类图片数量地平衡。

## 3.调参优化策略
1. OCRNet1使用sgd优化器，PolynomialDecay多项式学习率衰减，初始学习率为0.00125，power为0.9。使用CrossEntropyLoss交叉熵损失函数。batch_size为8，迭代次数为20000。
2. OCRNet2的策略与OCRNet1的策略相同。
3. OCRNet3的策略与OCRNet1的策略的不同之处在于损失函数，OCRNet3的损失函数为BootstrappedCrossEntropyLoss以及DiceLoss，DiceLoss是一种用于度量两个样本相似性的函数，其计算过程与IOU类似。
4. Deeplabv3p的策略与OCRNet1的策略相同。

## 4.训练脚本
示例：  

python train.py \
       --config ../configs/ocrnet1.yml \
       --do_eval \
       --use_vdl \
       --save_interval 5000 \
       --save_dir ../model/ocrnet1  
       
训练日志见/home/aistudio/work/log/train_log.txt

## 5.测试脚本
示例：

python val.py \
       --config ../configs/ocrnet1.yml \
       --model_path ../model/ocrnet1/best_model/model.pdparams
       
评估日志见  
1. /home/aistudio/work/log/eval_log1.txt
2. /home/aistudio/work/log/eval_log2.txt
3. /home/aistudio/work/log/eval_log3.txt
4. /home/aistudio/work/log/eval_log4.txt  

可以看到4个模型中ocrnet1的前景iou为0.8173，ocrnet2的前景iou为0.8154，ocrnet3的前景iou为0.8201,deeplabv3的前景iou在之前的评估中为0.82左右，但是日志中显示为0.8624，这是因为在整理代码时重新划分了数据集，与之前的划分不同，导致评估时用的数据中有些数据可能是之前训练所用的训练集中的数据。此外，可以看到ocrnet3中使用的loss能够带来一定的效果的提升。

# 五、总结
本方案在PaddleSeg的基础上使用了三个OCRNet+HRNet_W48，一个Deeplabv3p+ResNet101_vd共四个模型进行融合。4个模型各自对输入图像进行预测得到4个结果，然后进行投票，即对4个结果取平均作为最终结果，最终结果中像素值越大，表示有越多模型认为该像素为前景。该方案在B榜的得分为0.77732。

感谢我的队友，感谢官方提供的学习机会，感谢Paddle，感谢官方的算力支持。
