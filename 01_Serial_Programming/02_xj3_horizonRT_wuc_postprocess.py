# Copyright (c) 2024，WuChao D-Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np
from scipy.special import softmax
from time import time
from hobot_dnn import pyeasy_dnn as dnn

img_path = "./test_img/zidane.jpg"
result_save_path = "./zidane.result.png"
quantize_model_path = "./yolov8s_detect_bernoulli2_640x640_NCHW_2cores.bin"
input_image_size = 640
conf=0.2
iou=0.5
conf_inverse = -np.log(1/conf - 1)
print("iou threshol = %.2f, conf threshol = %.2f"%(iou, conf))
print("sigmoid_inverse threshol = %.2f"%conf_inverse)


# 一些常量或函数
coco_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", 
    "bus", "train", "truck", "boat", "traffic light", 
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", 
    "cat", "dog", "horse", "sheep", "cow", 
    "elephant", "bear", "zebra", "giraffe", "backpack", 
    "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", 
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
    "wine glass", "cup", "fork", "knife", "spoon", 
    "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", 
    "cake", "chair", "couch", "potted plant", "bed", 
    "dining table", "toilet", "tv", "laptop", "mouse", 
    "remote", "keyboard", "cell phone", "microwave", "oven", 
    "toaster", "sink", "refrigerator", "book", "clock", 
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

yolo_colors = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]

def draw_detection(img, box, score, class_id):
    x1, y1, x2, y2 = box
    color = yolo_colors[class_id%20]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    label = f"{coco_names[class_id]}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    label_x = x1
    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

    # Draw a filled rectangle as the background for the label text
    cv2.rectangle(
        img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
    )

    # Draw the label text on the image
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

# 读取horizon_quantize模型, 并打印这个horizon_quantize模型的输入输出Tensor信息
begin_time = time()
quantize_model = dnn.load(quantize_model_path)
print("\033[0;31;40m" + "Load horizon quantize model time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")

print("-> input tensors")
for i, quantize_input in enumerate(quantize_model[0].inputs):
    print(f"intput[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

print("-> output tensors")
for i, quantize_input in enumerate(quantize_model[0].outputs):
    print(f"output[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

# 准备一些常量
# 提前将反量化系数准备好
s_bboxes_scale = quantize_model[0].outputs[0].properties.scale_data[:,np.newaxis]
m_bboxes_scale = quantize_model[0].outputs[1].properties.scale_data[:,np.newaxis]
l_bboxes_scale = quantize_model[0].outputs[2].properties.scale_data[:,np.newaxis]
s_clses_scale = quantize_model[0].outputs[3].properties.scale_data[np.newaxis, :, np.newaxis, np.newaxis]
m_clses_scale = quantize_model[0].outputs[4].properties.scale_data[np.newaxis, :, np.newaxis, np.newaxis]
l_clses_scale = quantize_model[0].outputs[5].properties.scale_data[np.newaxis, :, np.newaxis, np.newaxis]

# DFL求期望的系数, 只需要生成一次
weights_static = np.array([i for i in range(16)]).astype(np.float32)[np.newaxis, :, np.newaxis]

# 提前准备一些索引, 只需要生成一次
static_index = np.arange(8400)

# anchors, 只需要生成一次
s_anchor = np.stack([np.tile(np.linspace(0.5, 79.5, 80), reps=80), 
                     np.repeat(np.arange(0.5, 80.5, 1), 80)], axis=0)
m_anchor = np.stack([np.tile(np.linspace(0.5, 39.5, 40), reps=40), 
                     np.repeat(np.arange(0.5, 40.5, 1), 40)], axis=0)
l_anchor = np.stack([np.tile(np.linspace(0.5, 19.5, 20), reps=20), 
                     np.repeat(np.arange(0.5, 20.5, 1), 20)], axis=0)

# 读取图片并利用resize的方式进行前处理
begin_time = time()
img = cv2.imread(img_path)
print("\033[0;31;40m" + "Read image time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")

begin_time = time()
input_tensor = img.copy()
print("\033[0;31;40m" + "Deep Copy image time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")

begin_time = time()
input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2RGB)
input_tensor = cv2.resize(input_tensor, (input_image_size, input_image_size), interpolation=cv2.INTER_NEAREST)
# input_tensor = np.array(input_tensor) / 255.0
input_tensor = np.transpose(input_tensor, (2, 0, 1))
input_tensor = np.expand_dims(input_tensor, axis=0)# .astype(np.float32)  # NCHW
print("\033[0;31;40m" + "Pre Process time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")
print(f"{input_tensor.shape = }")

img_h, img_w = img.shape[0:2]
y_scale, x_scale = img_h/input_image_size, img_w/input_image_size

# 推理
begin_time = time()
quantize_outputs = quantize_model[0].forward(input_tensor)
print("\033[0;31;40m" + "Forward time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")

# 转为numpy
begin_time = time()
s_bboxes = quantize_outputs[0].buffer
m_bboxes = quantize_outputs[1].buffer
l_bboxes = quantize_outputs[2].buffer
s_clses = quantize_outputs[3].buffer
m_clses = quantize_outputs[4].buffer
l_clses = quantize_outputs[5].buffer

# classify 分支反量化
s_clses = s_clses.astype(np.float32) * s_clses_scale
m_clses = m_clses.astype(np.float32) * m_clses_scale
l_clses = l_clses.astype(np.float32) * l_clses_scale

# reshape
s_clses = s_clses[0].reshape(80, -1)
m_clses = m_clses[0].reshape(80, -1)
l_clses = l_clses[0].reshape(80, -1)
s_bboxes = s_bboxes[0].reshape(64, -1)
m_bboxes = m_bboxes[0].reshape(64, -1)
l_bboxes = l_bboxes[0].reshape(64, -1)

# 利用numpy向量化操作完成阈值筛选（优化版）
s_static_index = np.arange(6400)
m_static_index = np.arange(1600)
l_static_index = np.arange(400)


s_class_ids = np.argmax(s_clses, axis=0)  # 针对8400行，挑选出80个分数中的最大值的索引
s_max_scores = s_clses[s_class_ids,s_static_index] # 使用最大值的索引索引相应的最大值
s_valid_indices = np.flatnonzero(s_max_scores >= conf_inverse)  # 得到大于阈值分数的索引，此时为小数字

m_class_ids = np.argmax(m_clses, axis=0)  # 针对8400行，挑选出80个分数中的最大值的索引
m_max_scores = m_clses[m_class_ids,m_static_index] # 使用最大值的索引索引相应的最大值
m_valid_indices = np.flatnonzero(m_max_scores >= conf_inverse)  # 得到大于阈值分数的索引，此时为小数字

l_class_ids = np.argmax(l_clses, axis=0)  # 针对8400行，挑选出80个分数中的最大值的索引
l_max_scores = l_clses[l_class_ids,l_static_index] # 使用最大值的索引索引相应的最大值
l_valid_indices = np.flatnonzero(l_max_scores >= conf_inverse)  # 得到大于阈值分数的索引，此时为小数字

# 利用筛选结果，索引分数值和id值
s_scores = s_max_scores[s_valid_indices]
s_ids = s_class_ids[s_valid_indices]

m_scores = m_max_scores[m_valid_indices]
m_ids = m_class_ids[m_valid_indices]

l_scores = l_max_scores[l_valid_indices]
l_ids = l_class_ids[l_valid_indices]

# 3个Classify分类分支：Sigmoid计算
s_scores = 1 / (1 + np.exp(-s_scores))
m_scores = 1 / (1 + np.exp(-m_scores))
l_scores = 1 / (1 + np.exp(-l_scores))

# 3个Bounding Box分支：反量化
s_bboxes_float32 = s_bboxes[:,s_valid_indices].astype(np.float32) * s_bboxes_scale
m_bboxes_float32 = m_bboxes[:,m_valid_indices].astype(np.float32) * m_bboxes_scale
l_bboxes_float32 = l_bboxes[:,l_valid_indices].astype(np.float32) * l_bboxes_scale

# 3个Bounding Box分支：dist2bbox（ltrb2xyxy）
s_ltrb_indices = np.sum(softmax(s_bboxes_float32.reshape(4, 16,-1), axis=1) * weights_static, axis=1)
s_anchor_indices = s_anchor[:,s_valid_indices]
s_x1y1 = s_anchor_indices - s_ltrb_indices[0:2]
s_x2y2 = s_anchor_indices + s_ltrb_indices[2:4]
s_dbboxes = np.vstack([s_x1y1, s_x2y2]).transpose(1,0)*8

m_ltrb_indices = np.sum(softmax(m_bboxes_float32.reshape(4, 16,-1), axis=1) * weights_static, axis=1)
m_anchor_indices = m_anchor[:,m_valid_indices]
m_x1y1 = m_anchor_indices - m_ltrb_indices[0:2]
m_x2y2 = m_anchor_indices + m_ltrb_indices[2:4]
m_dbboxes = np.vstack([m_x1y1, m_x2y2]).transpose(1,0)*16

l_ltrb_indices = np.sum(softmax(l_bboxes_float32.reshape(4, 16,-1), axis=1) * weights_static, axis=1)
l_anchor_indices = l_anchor[:,l_valid_indices]
l_x1y1 = l_anchor_indices - l_ltrb_indices[0:2]
l_x2y2 = l_anchor_indices + l_ltrb_indices[2:4]
l_dbboxes = np.vstack([l_x1y1, l_x2y2]).transpose(1,0)*32


# 大中小特征层阈值筛选结果拼接
dbboxes = np.concatenate((s_dbboxes, m_dbboxes, l_dbboxes), axis=0)
scores = np.concatenate((s_scores, m_scores, l_scores), axis=0)
ids = np.concatenate((s_ids, m_ids, l_ids), axis=0)

# nms
indices = cv2.dnn.NMSBoxes(dbboxes, scores, conf, iou)
print("\033[0;31;40m" + "Post Process time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")


# 绘制
begin_time = time()
for index in indices:
    score = scores[index]
    class_id = ids[index]
    x1, y1, x2, y2 = dbboxes[index]
    x1, y1, x2, y2 = int(x1*x_scale), int(y1*y_scale), int(x2*x_scale), int(y2*y_scale)
    print("(%d, %d, %d, %d) -> %s: %.2f"%(x1,y1,x2,y2, coco_names[class_id], score))
    draw_detection(img, (x1, y1, x2, y2), score, class_id)
print("\033[0;31;40m" + "Draw Result time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")

# 保存图片到本地
begin_time = time()
cv2.imwrite(result_save_path, img)
print("\033[0;31;40m" + "cv2.imwrite time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")
