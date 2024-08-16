import argparse
import os
import shutil
import sys
from glob import glob

import numpy as np

sys.path.append(".")
from src.data import collect_paint_imgs, create_dataset_classify_paint
from src.model.classify import pred_classify, train_classify
from src.utils import json_handler

VER = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_type", type=str, help="'label' or 'label_type'")

    # optional
    parser.add_argument(
        "-cd", "--create_dataset", required=False, action="store_true", default=False
    )
    parser.add_argument(
        "-tr", "--train", required=False, action="store_true", default=False
    )
    parser.add_argument("-v", "--version", required=False, type=int, default=None)
    args = parser.parse_args()

    data_type = args.data_type
    th_iou = args.th_iou
    th_sec = args.th_sec

    data_name = "classify_paint"
    data_root = f"datasets/{data_name}"
    os.makedirs(data_root, exist_ok=True)

    # create dataset
    if args.create_dataset:
        ann_json = json_handler.load("annotation/annotation.json")
        info_json = json_handler.load("annotation/info.json")
        data = collect_paint_imgs(ann_json, info_json)
        np.random.seed(42)
        random_idxs = np.random.choice(np.arange(len(data)), len(data))

        train_length = int(len(data) * 0.7)
        train_idxs = random_idxs[:train_length]
        test_idxs = random_idxs[train_length:]

        create_dataset_classify_paint(data, train_idxs, data_root, data_type, "train")
        create_dataset_classify_paint(data, test_idxs, data_root, data_type, "test")

    if args.train:
        # train YOLO
        yolo_result_dir = train_classify(data_name, data_type)
    else:
        # only prediction
        v_num = args.version
        yolo_result_dir = f"runs/v{VER}/{data_name}/{data_type}"
        if v_num is not None:
            yolo_result_dir += f"-v{v_num}"

    # prediction
    train_paths = glob(
        os.path.join(
            f"datasets/v{VER}/{data_name}",
            data_type,
            "train",
            "**",
            "*.jpg",
        )
    )
    test_paths = glob(
        os.path.join(
            f"datasets/v{VER}/{data_name}",
            data_type,
            "test",
            "**",
            "*.jpg",
        )
    )

    results_train, missed_img_path_train = pred_classify(
        train_paths, "train", yolo_result_dir
    )
    results_test, missed_img_path_test = pred_classify(
        test_paths, "test", yolo_result_dir
    )

    missed_imgs_dir = os.path.join(yolo_result_dir, "missed_images_test")
    if os.path.exists(missed_imgs_dir):
        shutil.rmtree(missed_imgs_dir)
    os.makedirs(missed_imgs_dir, exist_ok=True)
    for path, label, pred_label in missed_img_path_test:
        img_name = os.path.basename(path)
        img_name = f"true-{label}_pred-{pred_label}_" + img_name
        move_path = os.path.join(missed_imgs_dir, img_name)
        shutil.copyfile(path, move_path)
