#!/bin/sh


INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=/Users/kehwaweng/Documents/ObjectDetection/create_custom_dataset/ssdlite_mobilenet_v2_few_sample.config
TRAINED_CKPT_PREFIX=/Users/kehwaweng/Documents/ObjectDetection/create_custom_dataset/save_model/model.ckpt-73807
OUTPUT_DIRECTORY=/Users/kehwaweng/Documents/ObjectDetection/create_custom_dataset/frozen_model/
# TRAINED_CKPT_PREFIX=/Users/kehwaweng/Documents/ObjectDetection/create_custom_dataset/save_model_r1_13/model.ckpt-25
# OUTPUT_DIRECTORY=/Users/kehwaweng/Documents/ObjectDetection/create_custom_dataset/frozen_model_r1_13/



# python /Users/kehwaweng/Documents/gdrive/test/models/research/object_detection/export_inference_graph.py \
python /Users/kehwaweng/Documents/gdrive/models/research/object_detection/export_inference_graph.py \
        --input_type=${INPUT_TYPE} \
        --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
        --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
        --output_directory=${OUTPUT_DIRECTORY}



# python object_detection/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=/Users/kehwaweng/Documents/ObjectDetection/create_custom_dataset/ssdlite_mobilenet_v2_few_sample.config --trained_checkpoint_prefix=/Users/kehwaweng/Documents/ObjectDetection/create_custom_dataset/save_model/model.ckpt-45012 --output_directory=/Users/kehwaweng/Documents/ObjectDetection/create_custom_dataset/frozen_model/
