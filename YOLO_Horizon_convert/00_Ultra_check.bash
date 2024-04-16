#!/usr/bin/env sh
# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

set -e -v
cd $(dirname $0) || exit

model_type="onnx"
onnx_model="no_dfl_3out_yolov8s.onnx"
# onnx_model="yolov8s_seg_origin.onnx"
# onnx_model="0_seg/yolov8s_seg_pt_modified.onnx"
march="bayes"

hb_mapper checker --model-type ${model_type} \
                  --model ${onnx_model} \
                  --march ${march}


# hb_mapper checker --model-type onnx --march bayes --model yolov8s_seg_origin.onnx
# hb_mapper checker --model-type onnx --march bayes --model yolov8s_seg_pt_modified_dboxtest.onnx
# hb_mapper checker --model-type onnx --march bernoulli2 --model no_dfl_3out_yolov8s.onnx