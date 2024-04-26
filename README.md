English| [简体中文](./README_cn.md)

# YOLOv8_HorizonRT
## Efficient Deployment of YOLOv8 Post-processing on XJ3 and XJ5

The following figure illustrates the performance of several computer vision detection tasks implemented using YOLOv8, with Pose and Segment both relying on the results from Detect. This document discusses the efficient implementation of post-processing for the Detect task.

<img src=".\img\(1)图像分类、(2)关键点检测、(3)物体检测和(4)实例分割举例.png" alt="(1) Image Classification, (2) Keypoint Detection, (3) Object Detection, and (4) Instance Segmentation Examples" style="zoom:48%;" />

The code is hosted on GitHub: <https://github.com/WuChao-2024/YOLOv8_HorizonRT.git>. If any Markdown math formulas display incorrectly, please refer to the repository on GitHub.

## 1. Introduction to the Public Model

The detection head and post-processing workflow of the public YOLOv8 model can be represented as follows. The Backbone and Neck components of YOLOv8 are efficiently accelerated by the BPU and thus omitted here. For details on modifications to these components for improved efficiency, see: <https://developer.horizon.cc/forumDetail/189779523032809473>

<img src=".\img\公版处理流程.png" alt="Public Model Processing Flow" style="zoom:12%;" />

In the Bounding box prediction branch, YOLOv8 - Detection employs a DFL structure, which provides 16 estimates for each bounding box edge based on anchor positions. These estimates undergo SoftMax computation followed by a convolution operation to calculate expectations, representing the core Anchor-Free design of YOLOv8 where each Grid Cell predicts a single Bounding box. Assuming 16 numbers ($l_p$ or $(t_p, t_p, b_p)$, with $p = 0, 1, ..., 15$) represent the offset of a particular edge, the calculation formula for the offset is:

$$
\hat{l} = \sum_{p=0}^{15}{\frac{p·e^{l_p}}{\sum_{q=0}^{15}{e^{l_q}}}}
$$

This computation is highly intensive, and quantization of the SoftMax operator may lead to precision degradation. Executing this part on the CPU using vectorized numpy implementation requires 50ms, making it a relatively time-consuming calculation.

Subsequently, the dist2bbox operation decodes the ltrb description of each Bounding Box into xyxy format. Ltrb represents the distances of the left, top, right, and bottom edges from the center of the corresponding Grid Cell. After converting relative positions to absolute ones and multiplying them by the sampling factors of the corresponding feature layers, the xyxy coordinates are obtained, denoting the predicted positions of the left-top and right-bottom corners of the Bounding Box.

<img src=".\img\ltrb2xyxy.jpg" alt="ltrb2xyxy" style="zoom:15%;" />

With an image input size of $Size=640$, for the $i^{th}$ feature map ($i=1, 2, 3$) in the Bounding box prediction branch, the downsample factor is denoted as $Stride(i)$. In YOLOv8, $Stride(1)=8$, $Stride(2)=16$, and $Stride(3)=32$, resulting in feature map sizes of $n_1 = 80$, $n_2 = 40$, and $n_3 = 20$, respectively. There are a total of $n_1^2 + n_2^2 + n_3^2 = 8400$ Grid Cells responsible for predicting 8400 Bounding Boxes.

For feature map $i$, the $x^{th}$ row and $y^{th}$ column predict the bounding box for the corresponding scale, where $x, y \in [0, n_i) \cap Z$ and $Z$ is the set of integers. After the DFL structure, the Bounding Box description is in ltrb format, while we require an xyxy description. The specific transformation relations are:

$$
x_1 = (x+0.5-l) \times Stride(i)
$$

$$
y_1 = (y+0.5-t) \times Stride(i)
$$

$$
x_2 = (x+0.5+r) \times Stride(i)
$$

$$
y_1 = (y+0.5+b) \times Stride(i)
$$

In the Classify category prediction branch, the categories of the 8400 potential Bounding Boxes are predicted. This data undergoes Sigmoid activation function calculations and is then filtered through a given threshold to identify potential targets, including their corresponding categories and scores. Finally, the xyxy values for the corresponding detection boxes are selected from the Bounding box prediction branch to detect the final targets, comprising category (id), score (score), and position (xyxy).

Lastly, NMS (Non-Maximum Suppression) is applied to eliminate duplicate recognized targets, yielding the final detection results.

## 2. Optimization Approach

The primary optimization approach involves processing only relevant data after determining qualified detection boxes, eliminating unnecessary dequantization and traversal computations. The trade-off point in the Classify branch concerns whether to perform the Sigmoid calculation on the BPU.

If the CPU computes the Sigmoid function, it yields very high float precision, and since the end of the ONNX model is a convolution operator, it can output detection results with high int32 precision. However, the dequantization coefficients for the scores (pre-activation) of the 80 categories within a single Grid Cell are different, necessitating $80 \times 8400$ traversal and dequantization computations to filter out a small number of results, saving subsequent computational steps.

If the BPU computes the Sigmoid function, the dequantization coefficients for the scores of the 80 categories within a single Grid Cell are identical, eliminating the need for $80 \times 8400$ traversal and dequantization computations. Compared to having the CPU compute Sigmoid, this saves a large-scale dequantization computation but may introduce precision loss risks.

This document opts for the CPU to compute Sigmoid. The optimized detection head and post-processing workflow can be depicted as follows:

<img src=".\img\优化处理流程.png" alt="Optimized Processing Flow" style="zoom:15%;" />

Assuming the score of a certain category within a Grid Cell is denoted as $x$, the integer data after activation is $y$, and the threshold for selection is denoted as $c$, the necessary and sufficient condition for this score to be qualified is:

$$
y = Sigmoid(x) = \frac{1}{1+e^{-x}} > c
$$

This leads to the necessary and sufficient condition:

$$
x > -ln\left(\frac{1}{C} - 1\right)
$$

Consequently, we can filter out $k_i$ potential targets in the Classify branch before dequantization, avoiding extensive Sigmoid computations on redundant data. After identifying these potential targets, we proceed to the Bounding box prediction branch to select the original data, performing dequantization, DFL, and dist2bbox operations on the limited amount of data to obtain the final results.

Unfortunately, the dequantization coefficients for the 80 categories within a single Grid Cell are all distinct, preventing the exploitation of the strictly monotonic increasing property of the composite function formed by the dequantization formula and the activation function to complete the filtering before dequantization. Nonetheless, since dequantization takes around 6ms, it is marginally acceptable. If the BPU computes this Sigmoid function and there is no precision risk, considering having the BPU compute it and perform filtering prior to dequantization to eliminate this extensive dequantization of data.

## 3. Data Measurement

Set the CPU to performance mode:

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
Test Data (referencing kite.jpg; reproduction available via source code in the GitHub repository)

| Detection Head or Post-processing Stage | Original / ms | Optimized / ms | Notes |
| --------------------------------------- | -------------- | -------------- | ------ |
| Preprocessing (resizing, nearest neighbor interpolation) | 5.36      | 5.83             | Unoptimized |
| BPU Inference                            | 11.74     | 11.46            | All dequantization nodes removed |
| hbDNN buffer to numpy array              | 6.02      | 6.12             | Unoptimized |
| 3 Classify Branches: Dequantization Time (numpy implementation) | 6.81      | 5.72             | Unoptimized |
| 3 Classify Branches: Reshape (numpy implementation)              | \         | 0.79             | Formerly concat and view operations |
| 3 Bounding Box Branches: Reshape (numpy implementation)              | Same      | 0.75             |                                            |
| Threshold-based Selection of Potential Targets (numpy implementation) | 26        | 17.39+1.15=18.54 | Formerly a for loop and python.list.append, slower; this is an initial numpy optimization |
| 3 Classify Branches: Sigmoid Computation (numpy implementation)       | 11.89     | 1.59             | Sigmoid computation performed on filtered small amount of data |
| 3 Bounding Box Branches: Dequantization Time (numpy implementation)    | 6.15      | 2.99             | Dequantization performed on filtered small amount of data |
| 3 Bounding Box Branches: dist2bbox (ltrb2xyxy, numpy implementation)  | 31        | 7.44             | Decoding performed on filtered small amount of data |
| Concatenation of Threshold-based Selection Results for Large, Medium, Small Feature Layers | \         | 1.81             |                                              |
| NMS (cv2.dnn.NMSBoxes implementation)                       | 1.392     | 1.487            | Unoptimized |

## Reference Process

Note: Carefully check for errors such as `No such file or directory` and do not copy and paste commands without examination.

1. Clone the ultralytics repository and refer to the YOLOv8 official documentation to configure the environment.
```Bash
$ git clone https://github.com/ultralytics/ultralytics.git
```
2. Navigate to the local repository and download the official pre-trained weights, using YOLOv8s-Detect model as an example.
```Bash
$ cd ultralytics
$ wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt
```
3. Uninstall command-line commands related to YOLO, enabling direct modification of the `./ultralytics/ultralytics` directory.
```Bash
$ conda list | grep ultralytics # or
$ pip list | grep ultralytics
$ pip uninstall ultralytics   # Uninstall
```
4. Modify the output header of Detect to separately output Bounding Box information and Classify information from the three feature layers, resulting in six output headers.

File path: `./ultralytics/ultralytics/nn/modules/head.py`, approximately line 43, replace the forward function of the Detect class with the following content:
```Python
def forward(self, x):# J5 Export
    bbox = []
    cls = []
    for i in range(self.nl):
        bbox.append(self.cv2[i](x[i]))
        cls.append(self.cv3[i](x[i]))
    return (bbox, cls)
```
File path: `./ultralytics/ultralytics/nn/modules/head.py`, approximately line 100, replace the forward function of the Segment class with the following content:
```python
def forward(self, x): # J5 Export
    p = self.proto(x[0])  # mask protos
    mc = [self.cv4[i](x[i]) for i in range(self.nl)]
    bbox, cls = self.detect(self, x)
    return (mc, bbox, cls, p) 
```
5. Run the following Python script. If encountering an error `No module named onnxsim`, install the missing module.
```Python
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
model.export(format='onnx', simplify=True, opset=11)
```
6. Refer to the Tian Gong Kai Wu toolchain manual and OE package documentation to inspect the model. Ensure all operators are on the BPU and proceed to compile.
```Python
(bpu) $ hb_mapper checker --model-type onnx --march bayes --model yolov8s.onnx
(bpu) $ hb_mapper makertbin --model-type onnx --config ./yolov8s.yaml
```
hb_mappe checker excerpted results:

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
... ...  All BPU operators  ... ...         
```
A complete yaml file will be provided in the appendix, with the following key parameters shown:

```YAML
model_parameters:
  march: "bayes"
  remove_node_type: "Dequantize"  # Remove all dequantization nodes
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
Copy the compiled bin model to the development board and use the hrt_model_exec tool to conduct performance testing, adjusting `thread_num` to explore the optimal thread count.

```Bash
hrt_model_exec perf --model_file yolov8s_640x640_NCHW.bin \
                --model_name="" \
                --core_id=0 \
                --frame_count=200 \
                --perf_time=0 \
                --thread_num=1 \
                --profile_path="."
```

## Serial Program Design Reference

Note: For Ubuntu 20.04 systems, refer to the RDK Ultra manual for flashing. The system comes with a Python interpreter and other development environments.

[1.2 System Flashing | RDK X3 User Manual (horizon.cc)](https://developer.horizon.cc/documents_rdk/installation/install_os)

Code is hosted on GitHub: https://github.com/WuChao-2024/YOLOv8_HorizonRT.git

Includes interactive Jupyter Notebook scripts and directly executable `.py` files, all running straightforwardly without complex encapsulation or nesting. Contains post-compilation bin models and runtime references for XJ3 and XJ5, under active maintenance.

Currently includes:
- Resolution 640, 80 classes
- Original Backbone + Neck ONNX and bin models, including Bernoulli2.
- Horizon-modified Backbone + Neck ONNX and bin models, incorporating Bernoulli2 and Bayes.

## Parallel Program Design Reference

Since Python does not perform any computational operations but merely calls BPU or numpy interfaces, it is an I/O-intensive program in terms of Python. Implementing multi-threading in Python is feasible and will not be affected by Python's global GIL lock. In the specific program design, please refer to the Github repository.

<img src=".\img\Parallel program.png" alt="Parallel program" style="zoom:15%;" />

## Application Scenarios and Reflections

This article attempts an operation akin to manually "pruning," wondering why YOLOv8 model inference involves so many unnecessary decoding computations. I speculate that during model training, in addition to forward propagation, backward propagation, differentiation, and gradient descent are also required, necessitating calculations for all 8400 Grid Cell Bounding Boxes to compute specific deviation values through the loss function, which are then propagated backward to update the entire network's weights. However, when deploying the model, only forward propagation is needed, making these computations dispensable; instead, we can select and process only relevant data.

The article also explores the idea of fusing dequantization nodes into the post-processing stage. This fusion approach is not simply about jointly iterating over data; rather, it leverages the simple properties (such as continuity, monotonicity, and boundedness) of the computation functions involved in the dequantization-to-thresholding筛选 process to perform early filtering, thereby eliminating unnecessary dequantizations.

## References

YOLOv8 Documentation: https://docs.ultralytics.com/

J5 Algorithm Toolchain Documentation: https://developer.horizon.cc/api/v1/fileData/horizon_j5_open_explorer_cn_doc/index.html

Implementation of Fused Dequantization Nodes: https://developer.horizon.cc/forumDetail/116476291842200072