import os
from pycocotools.coco import COCO
import argparse


def create(args):
    coco = COCO(annotation_file = args.annotation_path)
    category_id_to_label = {}
    for key in coco.cats.keys():
        id_ = coco.cats[key]["id"]
        name = coco.cats[key]["name"]
        category_id_to_label[id_] = name

    with open(args.output , "w") as writefile:
        for key, val in category_id_to_label.items():
            print("item {", file = writefile)
            print(f"  id: {key}", file = writefile)
            print(f"  name: '{val}'", file = writefile)
            print("}", file = writefile)


def parse_args():
    parser = argparse.ArgumentParser(description='Create pbtxt for tensorflow research object detection')

    parser.add_argument("--annotation-path",
                        "-ap",
                        type = str,
                        default = "./custom_dataset.json",
                        help = "json file with coco dataset format")

    parser.add_argument("--output",
                        "-o",
                        type = str,
                        default = "./label.pbtxt",
                        help = "output tfrecord path ")

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    create(args)