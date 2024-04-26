#!/user/bin/env python

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


# -*- coding:utf-8 -*-
# Author: WuChao D-Robotics
# Date: 2024-04-16
# Description: Parallel Programming with args

import cv2, argparse, sys
import numpy as np
from threading import Thread, Lock
from queue import Queue

from scipy.special import softmax
from time import time, sleep
from hobot_dnn import pyeasy_dnn as dnn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='YOLO_horizon_bin_model_zoo/yolov8s_detect_bayes_640x640_NCHW.bin', help='Path to Horizon BPU Quantized *.bin Model.\nxj3 (bernoulli2) or xj5 (bayes)')
    parser.add_argument('--test-img', type=str, default='./test_img/kite.jpg', help='Path to Load Test Image.')
    parser.add_argument('--classes-num', type=int, default=80, help='Classes Num to Detect.')
    parser.add_argument('--input-size', type=int, default=640, help='Model Input Size')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold.')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold.')
    opt = parser.parse_args()

    # 线程数配置
    n2 = 2  # 推理视频帧线程数

    # 用于控制线程的全局变量
    global is_loading, is_forwarding, is_writing
    is_loading, is_forwarding, is_writing = True, True, True

    # 推理实例
    model = YOLOv8_Detect_img(opt)


    # 任务队列
    global task_queue, save_queue
    task_queue = Queue(30)
    save_queue = Queue(30)  # 结果保存队列多缓存一些

    sleep(1)

    # 创建并启动读取线程
    video_path = "/dev/video0"
    task_loader = Dataloader_videoCapture(video_path, task_queue, 0.005)
    task_loader.start()

    # 创建并启动推理线程
    inference_threads = [InferenceThread(_, model, task_queue, save_queue, 0.005) for _ in range(n2)]
    for t in inference_threads:
        t.start()
    
    # 创建并启动结果保存线程
    result_writer = videoWriter_videoCapture(save_queue, 0.001)
    result_writer.start()

    # 创建并启动日志打印线程
    result_writer = msg_printer(task_queue, save_queue, 0.5)
    result_writer.start()    

    print("[INFO] wait_join")
    task_loader.join()
    for t in inference_threads:
        t.join()
    result_writer.join()
    result_writer.join()
    print("[INFO] All task done.")
    exit()



class YOLOv8_Detect_img():
    def __init__(self, opt):
        self.img_path = opt.test_img
        self.result_save_path = opt.test_img + ".result.png"
        self.quantize_model_path = opt.model_path
        self.input_image_size = opt.input_size
        self.classes_num = opt.classes_num
        self.conf=opt.conf_thres
        self.iou=opt.iou_thres
        self.conf_inverse = -np.log(1/self.conf - 1)
        print("iou threshol = %.2f, conf threshol = %.2f"%(self.iou, self.conf))
        print("sigmoid_inverse threshol = %.2f"%self.conf_inverse)

        ### 读取horizon_quantize模型, 并打印这个horizon_quantize模型的输入输出Tensor信息
        try:
            begin_time = time()
            self.quantize_model = dnn.load(self.quantize_model_path)
            print("\033[0;31;40m" + "Load horizon quantize model time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")
        except:
            print("❌ [Load Model Failed, Please check xj3 (bernoulli2) or xj5 (bayes).] ❌")
            exit()
        print("-> input tensors")
        for i, quantize_input in enumerate(self.quantize_model[0].inputs):
            print(f"intput[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        print("-> output tensors")
        for i, quantize_input in enumerate(self.quantize_model[0].outputs):
            print(f"output[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        ### 解码映射的顺序, 保证outpus[outputs_order[i]] = outputs_shape_order[i]
        # Large, Medium, Small 
        self.outputs_order = [-1,-1,-1,-1,-1,-1]
        outputs_shape_order = [(1, 64, self.input_image_size//8, self.input_image_size//8), # s_bbox
                            (1, 64, self.input_image_size//16, self.input_image_size//16), # m_bbox
                            (1, 64, self.input_image_size//32, self.input_image_size//32), # l_bbox
                            (1, self.classes_num, self.input_image_size//8, self.input_image_size//8), # s_cls
                            (1, self.classes_num, self.input_image_size//16, self.input_image_size//16), # m_cls
                            (1, self.classes_num, self.input_image_size//32, self.input_image_size//32)  # l_cls
                            ]
        for i in range(6):
            a, b = self.quantize_model[0].outputs[i].properties.shape[1:3]
            if a == 64:
                if b == self.input_image_size//8:
                    self.outputs_order[0] = i
                elif b == self.input_image_size//16:
                    self.outputs_order[1] = i
                elif b == self.input_image_size//32:
                    self.outputs_order[2] = i
            elif a == self.classes_num:
                if b == self.input_image_size//8:
                    self.outputs_order[3] = i
                elif b == self.input_image_size//16:
                    self.outputs_order[4] = i
                elif b == self.input_image_size//32:
                    self.outputs_order[5] = i
        for i in range(6):
            print(f"horizon_model.outputs[ outputs_order[{i}] ] = {self.quantize_model[0].outputs[self.outputs_order[i]].properties.shape}")
        if -1 in self.outputs_order:
            print("Sorry, your model don't match the post process code.")
        
        ### 准备一些常量
        # 提前将反量化系数准备好
        self.s_bboxes_scale = self.quantize_model[0].outputs[self.outputs_order[0]].properties.scale_data[:,np.newaxis]
        self.m_bboxes_scale = self.quantize_model[0].outputs[self.outputs_order[1]].properties.scale_data[:,np.newaxis]
        self.l_bboxes_scale = self.quantize_model[0].outputs[self.outputs_order[2]].properties.scale_data[:,np.newaxis]
        self.s_clses_scale = self.quantize_model[0].outputs[self.outputs_order[3]].properties.scale_data[np.newaxis, :, np.newaxis, np.newaxis]
        self.m_clses_scale = self.quantize_model[0].outputs[self.outputs_order[4]].properties.scale_data[np.newaxis, :, np.newaxis, np.newaxis]
        self.l_clses_scale = self.quantize_model[0].outputs[self.outputs_order[5]].properties.scale_data[np.newaxis, :, np.newaxis, np.newaxis]

        # DFL求期望的系数, 只需要生成一次
        self.weights_static = np.array([i for i in range(16)]).astype(np.float32)[np.newaxis, :, np.newaxis]

        # 提前准备一些索引, 只需要生成一次
        self.static_index = np.arange(8400)
        self.s_static_index = np.arange(6400)
        self.m_static_index = np.arange(1600)
        self.l_static_index = np.arange(400)

        # anchors, 只需要生成一次
        self.s_anchor = np.stack([np.tile(np.linspace(0.5, 79.5, 80), reps=80), 
                            np.repeat(np.arange(0.5, 80.5, 1), 80)], axis=0)
        self.m_anchor = np.stack([np.tile(np.linspace(0.5, 39.5, 40), reps=40), 
                            np.repeat(np.arange(0.5, 40.5, 1), 40)], axis=0)
        self.l_anchor = np.stack([np.tile(np.linspace(0.5, 19.5, 20), reps=20), 
                            np.repeat(np.arange(0.5, 20.5, 1), 20)], axis=0)

    def forward(self, input_tensor):
        return self.quantize_model[0].forward(input_tensor)

    def preprocess(self, input_tensor):
        # 利用resize的方式进行前处理 
        input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2RGB)
        input_tensor = cv2.resize(input_tensor, (self.input_image_size, self.input_image_size), interpolation=cv2.INTER_NEAREST)
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)# .astype(np.float32)  # NCHW

        return input_tensor
     
    def postprocess(self, quantize_outputs):
        # 转为numpy
        begin_time = time()
        s_bboxes = quantize_outputs[self.outputs_order[0]].buffer
        m_bboxes = quantize_outputs[self.outputs_order[1]].buffer
        l_bboxes = quantize_outputs[self.outputs_order[2]].buffer
        s_clses = quantize_outputs[self.outputs_order[3]].buffer
        m_clses = quantize_outputs[self.outputs_order[4]].buffer
        l_clses = quantize_outputs[self.outputs_order[5]].buffer

        # classify 分支反量化
        s_clses = s_clses.astype(np.float32) * self.s_clses_scale
        m_clses = m_clses.astype(np.float32) * self.m_clses_scale
        l_clses = l_clses.astype(np.float32) * self.l_clses_scale

        # reshape
        s_clses = s_clses[0].reshape(80, -1)
        m_clses = m_clses[0].reshape(80, -1)
        l_clses = l_clses[0].reshape(80, -1)
        s_bboxes = s_bboxes[0].reshape(64, -1)
        m_bboxes = m_bboxes[0].reshape(64, -1)
        l_bboxes = l_bboxes[0].reshape(64, -1)

        # 利用numpy向量化操作完成阈值筛选（优化版）
        s_class_ids = np.argmax(s_clses, axis=0)  # 针对8400行，挑选出80个分数中的最大值的索引
        s_max_scores = s_clses[s_class_ids, self.s_static_index] # 使用最大值的索引索引相应的最大值
        s_valid_indices = np.flatnonzero(s_max_scores >= self.conf_inverse)  # 得到大于阈值分数的索引，此时为小数字

        m_class_ids = np.argmax(m_clses, axis=0)  # 针对8400行，挑选出80个分数中的最大值的索引
        m_max_scores = m_clses[m_class_ids, self.m_static_index] # 使用最大值的索引索引相应的最大值
        m_valid_indices = np.flatnonzero(m_max_scores >= self.conf_inverse)  # 得到大于阈值分数的索引，此时为小数字

        l_class_ids = np.argmax(l_clses, axis=0)  # 针对8400行，挑选出80个分数中的最大值的索引
        l_max_scores = l_clses[l_class_ids, self.l_static_index] # 使用最大值的索引索引相应的最大值
        l_valid_indices = np.flatnonzero(l_max_scores >= self.conf_inverse)  # 得到大于阈值分数的索引，此时为小数字

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
        s_bboxes_float32 = s_bboxes[:,s_valid_indices].astype(np.float32) * self.s_bboxes_scale
        m_bboxes_float32 = m_bboxes[:,m_valid_indices].astype(np.float32) * self.m_bboxes_scale
        l_bboxes_float32 = l_bboxes[:,l_valid_indices].astype(np.float32) * self.l_bboxes_scale

        # 3个Bounding Box分支：dist2bbox(ltrb2xyxy)
        s_ltrb_indices = np.sum(softmax(s_bboxes_float32.reshape(4, 16,-1), axis=1) * self.weights_static, axis=1)
        s_anchor_indices = self.s_anchor[:,s_valid_indices]
        s_x1y1 = s_anchor_indices - s_ltrb_indices[0:2]
        s_x2y2 = s_anchor_indices + s_ltrb_indices[2:4]
        s_dbboxes = np.vstack([s_x1y1, s_x2y2]).transpose(1,0)*8

        m_ltrb_indices = np.sum(softmax(m_bboxes_float32.reshape(4, 16,-1), axis=1) * self.weights_static, axis=1)
        m_anchor_indices = self.m_anchor[:,m_valid_indices]
        m_x1y1 = m_anchor_indices - m_ltrb_indices[0:2]
        m_x2y2 = m_anchor_indices + m_ltrb_indices[2:4]
        m_dbboxes = np.vstack([m_x1y1, m_x2y2]).transpose(1,0)*16

        l_ltrb_indices = np.sum(softmax(l_bboxes_float32.reshape(4, 16,-1), axis=1) * self.weights_static, axis=1)
        l_anchor_indices = self.l_anchor[:,l_valid_indices]
        l_x1y1 = l_anchor_indices - l_ltrb_indices[0:2]
        l_x2y2 = l_anchor_indices + l_ltrb_indices[2:4]
        l_dbboxes = np.vstack([l_x1y1, l_x2y2]).transpose(1,0)*32


        # 大中小特征层阈值筛选结果拼接
        dbboxes = np.concatenate((s_dbboxes, m_dbboxes, l_dbboxes), axis=0)
        scores = np.concatenate((s_scores, m_scores, l_scores), axis=0)
        ids = np.concatenate((s_ids, m_ids, l_ids), axis=0)

        # nms
        indices = cv2.dnn.NMSBoxes(dbboxes, scores, self.conf, self.iou)

        return dbboxes, scores, ids, indices


def signal_handler(signal, frame):
    global is_loading, is_forwarding, is_writing
    is_loading, is_forwarding, is_writing = False, False, False
    print('Ctrl+C, stopping!!!')
    sys.exit(0)


class Dataloader_videoCapture(Thread):
    # 从cap中读帧, 一直读到无帧可读
    # delay_time 用于控制读帧的频率，尽量和极限帧率的帧间隔一致, 一般设置为0.033 s
    def __init__(self, video_path, task_queue, delay_time):
        Thread.__init__(self)
        self.cap = cv2.VideoCapture(video_path)
        self.task_queue = task_queue
        self.delay_time = delay_time
    def run(self):
        global is_loading
        while is_loading:
            if not self.task_queue.full():
                # begin_time = time()
                ret, frame = self.cap.read()
                if ret:
                    self.task_queue.put(frame)
                else:
                    is_loading = False
                    self.cap.release()
                    break
                # print("\033[0;31;40m" + "Read time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")            
            sleep(self.delay_time)
        print("[INFO] Dateloader thread exit.")

class InferenceThread(Thread):
    # 推理的线程
    def __init__(self, i, model, task_queue, result_queue, delay_time):
        Thread.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.model = model
        self.delay_time = delay_time
        self.i = i
    def run(self):
        global is_forwarding, is_running, frame_counter
        while is_forwarding:
            if not self.task_queue.empty():
                # begin_time = time()
                # 从任务队列取图
                img = self.task_queue.get()
                # 存储拉伸量
                img_h, img_w = img.shape[0:2]
                y_scale, x_scale = img_h/self.model.input_image_size, img_w/self.model.input_image_size
                # 前处理
                input_tensor = self.model.preprocess(img)
                # 推理
                output_tensors = self.model.forward(input_tensor)
                # 后处理
                dbboxes, scores, ids, indices = self.model.postprocess(output_tensors)
                # 绘制
                for index in indices:
                    score = scores[index]
                    class_id = ids[index]
                    x1, y1, x2, y2 = dbboxes[index]
                    x1, y1, x2, y2 = int(x1*x_scale), int(y1*y_scale), int(x2*x_scale), int(y2*y_scale)
                    draw_detection(img, (x1, y1, x2, y2), score, class_id)
                # 结果存入结果队列
                trytimes = 5
                while trytimes > 0:
                    trytimes -= 1
                    if not self.result_queue.full():
                        trytimes = 0
                        self.result_queue.put(img)
                # 帧率计数器自增
                if trytimes >= 0:
                    frame_counter += 1
                # print("\033[0;31;40m" + "Forward time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")
            elif not is_loading:
                is_forwarding = False
            sleep(self.delay_time)
        print(f"[INFO] Forward thread {self.i} exit.")

class videoWriter_videoCapture(Thread):
    # 从结果队列中读帧
    def __init__(self, result_queue, delay_time):
        Thread.__init__(self)
        self.result_queue = result_queue
        self.delay_time = delay_time
    def run(self):
        global is_forwarding, is_writing
        while is_writing:
            if not self.result_queue.empty():
                # begin_time = time()
                img_result = self.result_queue.get()
                cv2.putText(img_result, '%.2f fps'%fps, (40, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow("result", img_result)
                cv2.waitKey(1)
                # print("\033[0;31;40m" + "Frame Write time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")            
            elif not is_forwarding:
                #self.out.release()
                is_writing = False
            sleep(self.delay_time)
        print("[INFO] VideoWriter thread exit.")

class msg_printer(Thread): 
    # 用于计算帧率的全局变量
    def __init__(self, task_queue, save_queue, delay_time):
        Thread.__init__(self)
        self.delay_time = delay_time
        self.task_queue = task_queue
        self.save_queue = save_queue
    def run(self):
        global frame_counter, fps
        frame_counter = 0
        fps = 0.0
        begin_time = time()
        while is_loading or is_forwarding or is_writing:
            delta_time = time() - begin_time
            fps = frame_counter/delta_time
            frame_counter = 0
            begin_time = time()       
            print("Smart FPS = %.2f, task_queue_size = %d, save_queue_size = %d"%(fps, self.task_queue.qsize(), self.save_queue.qsize())) 
            sleep(self.delay_time)
        print("[INFO] msg_printer thread exit.")
  
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



if __name__ == "__main__":
    main()