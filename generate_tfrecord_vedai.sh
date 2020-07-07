#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
# Assumes COCO dataset have been downloaded tensorflow object detection

set -e
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/extras/CUPTI/lib64:/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda10.0
echo "Utilising CUDA 10.0..."

# Define base directory
CURRENT_DIR=$(pwd)
VEDAI_DIR="${CURRENT_DIR}"

# Create the output directories.
VEDAI_TRAIN_DIR="${CURRENT_DIR}/train_objdetect"
TF_RECORDS="${VEDAI_DIR}/tf_records"
TF_RECORDS_TRAIN="${VEDAI_TRAIN_DIR}/tf_records/vedai_train.record"
TF_RECORDS_VAL="${VEDAI_TRAIN_DIR}/tf_records/vedai_val.record"

#Remove old directory and create new directory with necessary files inside
if [ -d "${TF_RECORDS}" ]
then
  rm -r ${TF_RECORDS}
fi

mkdir -p "${TF_RECORDS}"
touch "${TF_RECORDS_TRAIN}"
touch "${TF_RECORDS_VAL}"

TRAIN_IMAGE_DIR="${VEDAI_DIR}/train"
VAL_IMAGE_DIR="${VEDAI_DIR}/validation"

TRAIN_ANNOTATION_DIR="${VEDAI_DIR}/annotations" 
TRAIN_ANNOTATIONS_FILE="${TRAIN_ANNOTATION_DIR}/train_vedai.csv"
VAL_ANNOTATIONS_FILE="${TRAIN_ANNOTATION_DIR}/val_vedai.csv"

# Build TFRecords of the image data.
cd "${CURRENT_DIR}"

#Using tensorflow provided script to generate tfrecord
python generate_tfrecord_vedai.py \
  --logtostderr \
  --include_masks \
  --train_image_dir="${TRAIN_IMAGE_DIR}" \
  --val_image_dir="${VAL_IMAGE_DIR}" \
  --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
  --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
  --output_path_train="${TF_RECORDS_TRAIN}" \
  --output_path_val="${TF_RECORDS_VAL}" \
  --type="CO"

