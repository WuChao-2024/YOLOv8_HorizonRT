# YOLOv8_HorizonRT
## XJ3, XJ5部署YOLOv8后处理的高效实现

下图表示了利用YOLOv8实现几种计算机视觉的检测任务的效果，其中Pose和Segment均以Detect的检测结果为基础，此文讨论Detect任务的后处理高效实现。

亮点：

 - 大量减少后处理无用数据的反量化，遍历和计算。
 - 所有的python for循环使用numpy向量化操作。
 - 模型尾部全为conv算子，全以int32高精度输出。

RDK X3 实测效果
<img src=".\img\YOLOV8s_Detect_xj3_demo.png" alt="YOLOV8s_Detect_xj3_demo.png" style="zoom:48%;" />

RDK Ultra 实测效果
<img src=".\img\YOLOV8s_Detect_xj5_demo.png" alt="YOLOV8s_Detect_xj5_demo.png" style="zoom:48%;" />

<img src=".\img\(1)图像分类、(2)关键点检测、(3)物体检测和(4)实例分割举例.png" alt="(1)图像分类、(2)关键点检测、(3)物体检测和(4)实例分割举例" style="zoom:48%;" />

代码托管在GitHub：https://github.com/WuChao-2024/YOLOv8_HorizonRT.git，如果Markdown数学公式显示异常可前往Github阅读。

## 1. 公版模型介绍

公版YOLOv8的检测头和后处理流程可用下图表示，其中YOLOv8的Backbone和Neck部分均可被BPU较好的加速，故在此处省略。另外关于Backbone和Neck部分的高效修改可参考：https://developer.horizon.cc/forumDetail/189779523032809473

<img src=".\img\公版处理流程.png" alt="公版处理流程" style="zoom:12%;" />

在Bounding box预测分支中，YOLOv8 - Detection应用了一个DFL结构，DFL结构会对每个框的某条边基于anchor的位置给出16个估计，对16个估计求SoftMax，然后通过一个卷积操作来求期望，这也是YOLOv8的Anchor Free的核心设计，即每个Grid Cell仅仅负责预测1个Bounding box。假设在对某一条边偏移量的预测中，这16个数字为 $ l_p $ 或者$ (t_p, t_p, b_p)$ ，其中$p = 0,1,...,15$那么偏移量的计算公式为：

$$
\hat{l} = \sum_{p=0}^{15}{\frac{p·e^{l_p}}{\sum_{q=0}^{15}{e^{l_q}}}}
$$


这个计算是相当密集的，并且其中SoftMax算子的量化可能会有精度下降问题。这部分放到CPU上，使用numpy向量化实现，对全部条边进行DFL结构的计算，需要50ms，是比较耗时的一个计算。

随后的dist2bbox操作，将每个Bounding Box的ltrb描述解码为xyxy描述，ltrb分别表示左上右下四条边距离相对于Grid Cell中心的距离，相对位置还原成绝对位置后，再乘以对应特征层的采样倍数，即可还原成xyxy坐标，xyxy表示Bounding Box的左上角和右下角两个点坐标的预测值。

<img src=".\img\ltrb2xyxy.jpg" alt="ltrb2xyxy" style="zoom:15%;" />

图片输入为$Size=640$，对于Bounding box预测分支的第$i$个特征图$(i=1, 2, 3)$，对应的下采样倍数记为$Stride(i)$，在YOLOv8中，$Stride(1)=8, Stride(2)=16, Stride(3)=32$，对应特征图的尺寸记为$n_i = {Size}/{Stride(i)}$，即尺寸为$n_1 = 80, n_2 = 40 ,n_3 = 20$三个特征图，一共有$n_1^2+n_2^2+n_3^3=8400$个Grid Cell，负责预测8400个Bounding Box。

对特征图$i$，第$x$行$y$列负责预测对应尺度Bounding Box的检测框，其中$x,y \in [0, n_i)\bigcap{Z}$，$Z$为整数的集合。DFL结构后的Bounding Box检测框描述为$ltrb$描述，而我们需要的是$xyxy$描述，具体的转化关系如下：
$$
x_1 = (x+0.5-l)\times{Stride(i)}
$$

$$
y_1 = (y+0.5-t)\times{Stride(i)}
$$

$$
x_2 = (x+0.5+r)\times{Stride(i)}
$$

$$
y_1 = (y+0.5+b)\times{Stride(i)}
$$

在Classify类别预测分支中，会对8400个潜在的Bounding Box的类别进行预测，这部分的数据会经过Sigmoid激活函数计算，再经过给定的阈值Threshold筛选出个潜在的目标，包括对应的类别和分数，这时候再对应的去Bounding box预测分支中Select对应检测框的xyxy值，即可检测到最终的目标，包括类别(id)，分数(score)和位置(xyxy)。

最后，通过nms操作，剔除重复识别的目标，得到最终的检测结果。



## 2. 优化思路

主要的优化思路在于确定对应检测框合格后，再对其相关数据进行处理，去掉无用数据的反量化和遍历计算。在Classify类别预测分支中，权衡点在于sigmoid计算是否让BPU计算。

如果让CPU计算Sigmoid，会得到非常高的float精度，并且onnx模型尾部就是一个卷积算子，可以以int32高精度输出检测结果。但是，对于同一个Grid Cell的80类别的分数(激活前)的反量化系数是不同的，所以必须先进行$80\times8400$次遍历和反量化计算，才能筛选出少量结果，节约其他的计算流程。

如果让BPU计算Sigmoid，对于同一个Grid Cell的80类别的分数的反量化系数是相同的，这样就不需要进行$80\times8400$次遍历和反量化计算，直接筛选出少量结果。相较于让CPU计算Sigmoid，节约了一次大规模的反量化计算，但是可能存在精度损失的风险。

本文选择让CPU计算Sigmoid，优化后的检测头和后处理流程可用下图表示。

<img src=".\img\优化处理流程.png" alt="优化处理流程" style="zoom:15%;" />

我们假设某一个Grid Cell的某一个类别的分数记为$x$，激活函数作用完的整型数据为$y$，阈值筛选的过程会给定一个阈值，记为$c$，那么此分数合格的充分必要条件为：
$$
y=Sigmoid(x)=\frac{1}{1+e^{-x}}>c
$$
由此可以得出此分数合格的充分必要条件为：
$$
x > -ln\left(\frac{1}{C}-1\right)
$$
综上所述，我们可以在Classify分类分支反量化前筛选出潜在的$k_i$个目标，避免大量无用数据的Sigmoid的计算。而在筛选出潜在的$k_i$个目标后，再去Bounding box预测分支的Select出原始的数据，针对有限个数据进行反量化，DFL和dist2bbox的操作，得到最终的结果。 

可惜对于同一个Grid Cell的80个类别的反量化系数均不相同，所以不能利用反量化公式和激活函数公式组成的复合函数严格单调增的性质，在反量化前就筛选完，避免掉大量数据的反量化操作，好在反量化也就占用6ms左右，勉强可以接受。如果让BPU计算这个Sigmoid函数，那么所有的反量化系数都是相同，没有精度风险时，可以考虑让BPU计算，然后反量化前进行筛选，避免掉这部分大量数据的反量化。


## 3. 数据实测

将CPU设置为性能模式

```bash
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy1/scaling_governor"
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy2/scaling_governor"
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy3/scaling_governor"
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor"
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy5/scaling_governor"
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor"
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy7/scaling_governor"
```

测试数据（参考kite.jpg图片，可参考github仓库源码复现）

| 检测头或者后处理过程                                   | 原版 / ms | 优化后 / ms      | 备注                                                         |
| ------------------------------------------------------ | --------- | ---------------- | ------------------------------------------------------------ |
| 前处理 (resize, 最近邻插值)                            | 5.36      | 5.83             | 未优化                                                       |
| BPU推理                                                | 11.74     | 11.46            | 均已去除所有反量化节点                                       |
| hbDNN buffer 2 numpy array                             | 6.02      | 6.12             | 未优化                                                       |
| 3个Classify分类分支：反量化耗时（numpy实现）           | 6.81      | 5.72             | 未优化                                                       |
| 3个Classify分类分支：Reshape（numpy实现）              | \         | 0.79             | 前者为concat和view等操作                                     |
| 3个Bounding Box分支：Reshape（numpy实现）              | 同上      | 0.75             |                                                              |
| 阈值筛选出潜在的目标（numpy实现）                      | 26        | 17.39+1.15=18.54 | 原版本来是for循环，和python.list.append操作，更慢，此处是numpy优化初版的数据。 |
| 3个Classify分类分支：Sigmoid计算（numpy实现）          | 11.89     | 1.59             | 对筛选完的少量数据进行Sigmoid计算                            |
| 3个Bounding Box分支：反量化耗时（numpy实现）           | 6.15      | 2.99             | 对筛选完的少量数据进行反量化                                 |
| 3个Bounding Box分支：dist2bbox（ltrb2xyxy, numpy实现） | 31        | 7.44             | 对筛选完的少量数据进行解码                                   |
| 大中小特征层阈值筛选结果拼接                           | \         | 1.81             |                                                              |
| NMS（cv2.dnn.NMSBoxes实现）                            | 1.392     | 1.487            | 未优化                                                       |





## 4. 流程参考

注：任何`No such file or directory`等报错请仔细检查，请勿逐条复制运行。

下载ultralytics仓库，并参考YOLOv8官方文档，配置好环境

```Bash
$ git clone https://github.com/ultralytics/ultralytics.git
```

进入本地仓库，下载官方的预训练权重，这里以YOLOv8s-Detect模型为例

```Bash
$ cd ultralytics
$ wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt
```

卸载yolo相关的命令行命令，这样直接修改`./ultralytics/ultralytics`目录即可生效。

```Bash
$ conda list | grep ultralytics # 或者
$ pip list | grep ultralytics
$ pip uninstall ultralytics   # 卸载
```

修改Detect的输出头，直接将三个特征层的Bounding Box信息和Classify信息分开输出，一共6个输出头。

文件目录：`./ultralytics/ultralytics/nn/modules/head.py`，约第43行，Detect类的forward函数替换成以下内容：

```Python
def forward(self, x):# J5 Export
    bbox = []
    cls = []
    for i in range(self.nl):
        bbox.append(self.cv2[i](x[i]))
        cls.append(self.cv3[i](x[i]))
    return (bbox, cls)
```

运行以下Python脚本，如果有`No module named onnxsim`报错，安装一个即可

```Python
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
model.export(format='onnx', simplify=True, opset=11)
```

参考天工开物工具链手册和OE包的参考，对模型进行检查，所有算子均在BPU上，进行编译即可：

```Python
(bpu) $ hb_mapper checker --model-type onnx --march bayes --model yolov8s.onnx
(bpu) $ hb_mapper makertbin --model-type onnx --config ./yolov8s.yaml
```

hb_mappe checker 结果（节选）

```Bash
2024-04-10 20:29:03,686 INFO hbdk version 3.46.4
2024-04-10 20:29:03,686 INFO horizon_nn version 0.19.3
2024-04-10 20:29:03,686 INFO hb_mapper version 1.18.2
... ...
2024-04-10 20:29:03,898 INFO Input ONNX model infomation:
ONNX IR version:          6
Opset version:            [11, 1]
Producer:                 pytorch2.1.1
Domain:                   none
Input name:               images, [1, 3, 640, 640]
Output name:              output0, [1, 64, 80, 80]
Output name:              331, [1, 64, 40, 40]
Output name:              345, [1, 64, 20, 20]
Output name:              324, [1, 80, 80, 80]
Output name:              338, [1, 80, 40, 40]
Output name:              352, [1, 80, 20, 20]
... ...
2024-04-10 20:29:13,449 INFO consumed time 0.741772
2024-04-10 20:29:13,578 INFO FPS=30.93, latency = 32328.6 us   (see ./.hb_check/main_graph_subgraph_0.html)
2024-04-10 20:29:13,785 INFO The converted model node information:
==========================================================================================
Node                                             ON   Subgraph  Type                       
-------------------------------------------------------------------------------------------
/model.0/conv/Conv                               BPU  id(0)     HzSQuantizedConv           
/model.0/act/Mul                                 BPU  id(0)     HzLut                  
/model.1/conv/Conv                               BPU  id(0)     HzSQuantizedConv           
/model.1/act/Mul                                 BPU  id(0)     HzLut                      
/model.2/cv1/conv/Conv                           BPU  id(0)     HzSQuantizedConv           
/model.2/cv1/act/Mul                             BPU  id(0)     HzLut                      
/model.2/Split                                   BPU  id(0)     Split                       
... ...  均为BPU算子  ... ...         
```

完整的yaml文件会在附录给出，这里给出关键的几个参数：

```YAML
model_parameters:
  march: "bayes"
  remove_node_type: "Dequantize"  # 移除所有的反量化节点
  ... ...
input_parameters:
  input_type_rt: 'rgb'    
  input_layout_rt: 'NCHW'
  input_type_train: 'rgb'
  input_layout_train: 'NCHW'
  norm_type: 'data_scale'
  scale_value: 0.003921568627451
  ... ...
calibration_parameters:
  ... ...
compiler_parameters:
  compile_mode: 'latency'
  optimize_level: 'O3'
  ... ...
```

将编译后的bin模型拷贝到开发板，使用hrt_model_exec工具进行性能实测，其中可以调整thread_num来试探最佳的线程数量。

```Bash
hrt_model_exec perf --model_file yolov8s_640x640_NCHW.bin \                      
                --model_name="" \                      
                --core_id=0 \                      
                --frame_count=200 \                      
                --perf_time=0 \                      
                --thread_num=1 \                      
                --profile_path="."
```



## 5. 串行程序设计参考

注：Ubuntu 20.04系统需参考RDK Ultra手册烧录，系统自带Python解释器等开发环境。

[1.2 系统烧录 | RDK X3用户手册 (horizon.cc)](https://developer.horizon.cc/documents_rdk/installation/install_os)

代码托管在GitHub：https://github.com/WuChao-2024/YOLOv8_HorizonRT.git

包括ipynb的交互式运行脚本和.py直接运行的py文件，均直白式运行，没有复杂的封装和嵌套，包含XJ3和XJ5的编译后bin模型和运行参考，长期维护中。
目前包括: 
640分辨率 80类别
原版Backbone + Neck的onnx与bin模型, 包括Bernoulli2。
地平线改Backbone + Neck的onnx与bin模型, 包括Bernoulli2和Bayes。

## 6. 并行程序设计参考
由于python不承担任何计算操作, 只是调用BPU或者numpy的接口, 所以对于Python程序而言是一个IO密集的程序, Python实现多线程可行, 且不会受到Pyhton全局GIL锁影响. 具体程序更新完善中, 请关注Github仓库. 目前, Python的多线程程序, YOLOv8s, Detect, 640×640分辨率, 80类别, 在X3上可跑到15fps, 在Ultra上可跑到30fps, 在批量的预测的精度验证的应用中可一定的加快速度, 未来会封装进TROS以进一步提高效率, 满足实时视频流目标检测的需求.

## 7. 应用场景和思考

本文尝试了类似于手搓“剪枝”的操作，为什么感觉YOLOv8的模型推理多了这么多无用的解码计算？我想应该是在模型训练时，不仅仅需要前向传播，还需要反向传播、求导和梯度下降，所以会对8400个Grid Cell的Bounding Box全部进行计算，从而通过损失函数计算出具体的偏差值，再反向传播，将偏差信息带给整个网络，从而更新网络的权重。而我们部署模型时，只需要前向传播，不需要反向传播，所以这部分计算可以省略，挑选有用的数据进行计算。

本文还尝试了反量化节点融合到后处理的一种思路，这种反量化融合后处理的方式并不是简单的将数据“一同遍历”，而是在于利用反量化到后处理中阈值筛选的所有过程的计算函数的简单性质（连续性，单调性，有界性等等）提前进行筛选，避免掉无用反量化。



## 参考

YOLOv8文档：https://docs.ultralytics.com/

J5算法工具链文档：https://developer.horizon.cc/api/v1/fileData/horizon_j5_open_explorer_cn_doc/index.html

反量化节点的融合实现：https://developer.horizon.cc/forumDetail/116476291842200072