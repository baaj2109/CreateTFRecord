import os
import argsparse

from pycocotools.coco import COCO
from PIL import Image
from random import shuffle
import os, sys
import numpy as np
import tensorflow as tf
import logging

import utils


def load_coco_dection_dataset(imgs_dir, annotations_filepath):
    """Load data from dataset by pycocotools. This tools can be download from "http://mscoco.org/dataset/#download"
    Args:
        imgs_dir: directories of coco images
        annotations_filepath: file path of coco annotations file
    Return:
        coco_data: list of dictionary format information of each image
    """
    coco = COCO(annotations_filepath)
    img_ids = coco.getImgIds() 
    cat_ids = coco.getCatIds() 

    shuffle(img_ids)

    coco_data = []

    nb_imgs = len(img_ids)
    for index, img_id in enumerate(img_ids):
        if index % 100 == 0:
            print("Readling images: %d / %d "%(index, nb_imgs))
        img_info = {}
        bboxes = []
        labels = []

        img_detail = coco.loadImgs(img_id)[0]
        pic_height = img_detail['height']
        pic_width = img_detail['width']

        ann_ids = coco.getAnnIds(imgIds=img_id,catIds=cat_ids)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            bboxes_data = ann['bbox']
            bboxes_data = [bboxes_data[0]/float(pic_width), bboxes_data[1]/float(pic_height),\
                                  bboxes_data[2]/float(pic_width), bboxes_data[3]/float(pic_height)]
                         # the format of coco bounding boxs is [Xmin, Ymin, width, height]
            bboxes.append(bboxes_data)
            labels.append(ann['category_id'])


        img_path = os.path.join(imgs_dir, img_detail['file_name'])
        img_bytes = tf.gfile.FastGFile(img_path,'rb').read()

        img_info['pixel_data'] = img_bytes
        img_info['height'] = pic_height
        img_info['width'] = pic_width
        img_info['bboxes'] = bboxes
        img_info['labels'] = labels

        coco_data.append(img_info)
    return coco_data





    def dict_to_coco_example(img_data):
    """Convert python dictionary formath data of one image to tf.Example proto.
    Args:
        img_data: infomation of one image, inclue bounding box, labels of bounding box,\
            height, width, encoded pixel data.
    Returns:
        example: The converted tf.Example
    """
    bboxes = img_data['bboxes']
    xmin, xmax, ymin, ymax = [], [], [], []
    for bbox in bboxes:
        xmin.append(bbox[0])
        xmax.append(bbox[0] + bbox[2])
        ymin.append(bbox[1])
        ymax.append(bbox[1] + bbox[3])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': utils.int64_feature(img_data['height']),
        'image/width': utils.int64_feature(img_data['width']),
        'image/object/bbox/xmin': utils.float_list_feature(xmin),
        'image/object/bbox/xmax': utils.float_list_feature(xmax),
        'image/object/bbox/ymin': utils.float_list_feature(ymin),
        'image/object/bbox/ymax': utils.float_list_feature(ymax),
        'image/object/class/label': utils.int64_list_feature(img_data['labels']),
        'image/encoded': utils.bytes_feature(img_data['pixel_data']),
        'image/format': utils.bytes_feature('jpeg'.encode('utf-8')),
    }))
    return example


def convert(args):

    coco_data = load_coco_dection_dataset(args.image_root, args.annotation_path)
    total_imgs = len(coco_data)
    # write coco data to tf record
    with tf.python_io.TFRecordWriter(args.output) as tfrecord_writer:
        for index, img_data in enumerate(coco_data):
            if index % 100 == 0:
                print("Converting images: %d / %d" % (index, total_imgs))
            example = dict_to_coco_example(img_data)
            tfrecord_writer.write(example.SerializeToString())




def parse_args():
    parser = argparse.ArgumentParser(description='Create tfrecord dataset for tensorflow research object detection')

    parser.add_argument("--image-root",
                        "-ir"
                        type = str,
                        default = "./image_root",
                        help = 'image folder root')
                        '''
                            image_root
                                    |
                                    --- chair
                                    --- couch
                                    --- table
                        '''

    parser.add_argument("--annotation-path",
                        "-ap"
                        type = str,
                        default = "./custom_dataset.json",
                        help = "json file with coco dataset format")

    parser.add_argument("--output",
                        "-o",
                        type = str,
                        default = "./custom_dataset.tfrecord",
                        help = "output tfrecord path ")

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    convert(args)










