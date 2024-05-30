import argparse
import os
import shutil
import sys
from glob import glob

import numpy as np
from ultralytics import YOLO

sys.path.append(".")
from src.utils import json_handler

image_paths = sorted(glob(os.path.join("annotation/images", "*.jpg")))
annotation_json = json_handler.load("annotation/annotation.json")
info_json = json_handler.load("annotation/info.json")


parser = argparse.ArgumentParser()
parser.add_argument(
    "-cd", "--create_dataset", required=False, action="store_true", default=False
)
args = parser.parse_args()


out_dir = "./datasets/classify_paint/"


def create_dataset(ann_data, idxs, out_dir, typ, stage):
    for idx in idxs:
        video_id, aid, label, label_type, img_name = ann_data[idx]

        if typ == "label":
            lbl_txt = label  # A11~C42
        elif typ == "label_type":
            lbl_txt = label_type  # only A, B, C
        else:
            raise ValueError

        img_path = os.path.join(out_dir, typ, stage, lbl_txt, img_name)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        shutil.copyfile(os.path.join("annotation/images", img_name), img_path)


if args.create_dataset:
    # collect data
    ann_data = []
    for image_path in image_paths:
        video_id, aid, _ = os.path.basename(image_path).split("_")
        if video_id not in info_json:
            # print("skip", video_id)
            continue

        ann_lst = [ann for ann in annotation_json[video_id] if ann["reply"] == aid]
        if len(ann_lst) == 0:
            print("not found reply", video_id, aid)
            continue
        elif len(ann_lst) > 1:
            print("duplicated", video_id, aid, image_path)

        for i, ann in enumerate(ann_lst):
            label = ann["text"]

            try:
                label = label.split("(")[1].replace(")", "")  # extract within bracket
            except IndexError:
                print("error label", video_id, aid, label, image_path)
                continue

            # if i > 0:
            #     print("success", video_id, aid, label)
            break
        else:
            continue

        label_type = label[:1]
        ann_data.append(
            (video_id, aid, label, label_type, os.path.basename(image_path))
        )

    # create yolo dataset
    np.random.seed(42)
    random_idxs = np.random.choice(np.arange(len(ann_data)), len(ann_data))

    train_length = int(len(ann_data) * 0.7)
    train_idxs = random_idxs[:train_length]
    test_idxs = random_idxs[train_length:]

    create_dataset(ann_data, train_idxs, out_dir, "label", "train")
    create_dataset(ann_data, test_idxs, out_dir, "label", "test")
    create_dataset(ann_data, train_idxs, out_dir, "label_type", "train")
    create_dataset(ann_data, test_idxs, out_dir, "label_type", "test")


# train YOLO
model = YOLO("yolov8n-cls.pt")
result = model.train(data="classify_paint/label_type/", epochs=100, task="classify")
