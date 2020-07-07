#!/bin/bash
set -e
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/extras/CUPTI/lib64:/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda10.0
echo "Utilising CUDA 10.0..."

# Define base directory
CURRENT_DIR=$(pwd)

# Config files
ssd_mobilenet_config_50k=./train_objdetect/ssd_mobilenet_v1_vedai_50k.config
ssd_mobilenet_config_200k=./train_objdetect/ssd_mobilenet_v1_vedai_200k.config
ssd_inceptionv2_config_200k=./train_objdetect/ssd_inception_v2_vedai_200k.config
faster_rcnn_resnet50_config_200k=./train_objdetect/faster_rcnn_resnet50_vedai_200k.config

# Checkpoint directories
ssd_mobilenet_output_50k=./train_objdetect/output_mobilenetv1_variant_50k
ssd_mobilenet_output_200k=./train_objdetect/output_mobilenetv1_variant_200k 
ssd_inceptionv2_output_200k=./train_objdetect/output_inceptionv2_variant_200k 
faster_rcnn_resnet50_output_200k=./train_objdetect/output_faster_rcnn_resnet50_variant_200k 

mobilenet_inference_graph_dir_50k=./mobilenet_50k
mobilenet_inference_graph_dir_200k=./mobilenet_200k
inception_inference_graph_dir_200k=./inception_200k
rcnn_resnet_inference_graph_dir_200k=./rcnn_resnet_200k

#Remove old directory and create new directory with necessary files inside
: '
if [ -d "$mobilenet_inference_graph_dir_50k" ]
then
  rm -r $mobilenet_inference_graph_dir_50k
fi
mkdir -p "$mobilenet_inference_graph_dir_50k"

if [ -d "$mobilenet_inference_graph_dir_200k" ]
then
  rm -r $mobilenet_inference_graph_dir_200k
fi
mkdir -p "$mobilenet_inference_graph_dir_200k"
'

if [ -d "$inception_inference_graph_dir_200k" ]
then
  rm -r $inception_inference_graph_dir_200k
fi
mkdir -p "$inception_inference_graph_dir_200k"

if [ -d "$rcnn_resnet_inference_graph_dir_200k" ]
then
  rm -r $rcnn_resnet_inference_graph_dir_200k
fi
mkdir -p "$rcnn_resnet_inference_graph_dir_200k"


#Using tensorflow provided script to generate tfrecord
: '
#Using mobilenet 50k
python export_inference_graph.py \
--input_type image_tensor \
--pipeline_config_path $ssd_mobilenet_config_50k \
--trained_checkpoint_prefix $ssd_mobilenet_output_50k/model.ckpt-50000 \
--output_directory $mobilenet_inference_graph_dir_50k

#Using mobilenet_200k
python export_inference_graph.py \
--input_type image_tensor \
--pipeline_config_path $ssd_mobilenet_config_200k \
--trained_checkpoint_prefix $sslsd_mobilenet_output_200k/model.ckpt-200000 \
--output_directory $mobilenet_inference_graph_dir_200k
'
#Using inceptionv2_200k
python export_inference_graph.py \
--input_type image_tensor \
--pipeline_config_path $ssd_inceptionv2_config_200k \
--trained_checkpoint_prefix $ssd_inceptionv2_output_200k/model.ckpt-200000 \
--output_directory $inception_inference_graph_dir_200k

#Using faster_rcnn_resnet50_200k
python export_inference_graph.py \
--input_type image_tensor \
--pipeline_config_path $faster_rcnn_resnet50_config_200k \
--trained_checkpoint_prefix $faster_rcnn_resnet50_output_200k/model.ckpt-81779 \
--output_directory $rcnn_resnet_inference_graph_dir_200k
'
