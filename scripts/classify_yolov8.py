import argparse
import os
import shutil
import sys
from glob import glob

import numpy as np
from tqdm import tqdm

sys.path.append(".")
from src.data import (
    collect_paint_imgs,
    create_dataset_classify_paint,
    create_dataset_classify_yolo_pred,
    extract_yolo_pred_imgs,
)
from src.model.classify import pred_classify, train_classify
from src.utils import json_handler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_type", type=str, help="'paint' or 'yolo'")
    parser.add_argument("data_type", type=str, help="'label' or 'label_type'")

    # optional(dataset_type == 'yolo')
    parser.add_argument("-iou", "--th_iou", required=False, type=float, default=0.1)
    parser.add_argument("-sec", "--th_sec", required=False, type=float, default=1.0)

    # optional
    parser.add_argument(
        "-cd", "--create_dataset", required=False, action="store_true", default=False
    )
    parser.add_argument(
        "-tr", "--train", required=False, action="store_true", default=False
    )
    parser.add_argument("-v", "--version", required=False, type=int, default=None)
    args = parser.parse_args()

    dataset_type = args.dataset_type
    data_type = args.data_type
    th_iou = args.th_iou
    th_sec = args.th_sec

    image_paths = sorted(glob(os.path.join("annotation/images", "*.jpg")))
    annotation_json = json_handler.load("annotation/annotation.json")
    info_json = json_handler.load("annotation/info.json")

    data_name = f"classify_{dataset_type}"
    if dataset_type == "yolo":
        data_name += f"_sec{th_sec}_iou{th_iou}"
    data_root = f"datasets/{data_name}"
    os.makedirs(data_root, exist_ok=True)

    if args.create_dataset:
        ann_json = json_handler.load("annotation/annotation.json")
        info_json = json_handler.load("annotation/info.json")
        if dataset_type == "paint":
            imgs = collect_paint_imgs(ann_json, info_json)
        elif dataset_type == "yolo":
            paint_data_json = json_handler.load("annotation/paint_bbox.json")
            video_id_to_name = {
                data[0]: data[1].split(".")[0]
                for data in np.loadtxt(
                    "annotation/annotation.tsv",
                    str,
                    delimiter="\t",
                    skiprows=1,
                    usecols=[1, 2],
                )
                if data[0] != "" and data[1] != ""
            }

            imgs = []
            for video_id, ann_lst in tqdm(ann_json.items(), ncols=100):
                if video_id not in info_json:
                    tqdm.write(f"{video_id} is not in info.json")
                    continue

                video_name = video_id_to_name[video_id]
                imgs += extract_yolo_pred_imgs(video_name, th_sec, th_iou)
        else:
            raise ValueError

        np.random.seed(42)
        random_idxs = np.random.choice(np.arange(len(imgs)), len(imgs))

        train_length = int(len(imgs) * 0.7)
        train_idxs = random_idxs[:train_length]
        test_idxs = random_idxs[train_length:]

        if dataset_type == "paint":
            create_dataset_classify_paint(
                imgs, train_idxs, data_root, data_type, "train"
            )
            create_dataset_classify_paint(imgs, test_idxs, data_root, data_type, "test")
        elif dataset_type == "yolo":
            create_dataset_classify_yolo_pred(
                imgs, train_idxs, data_root, data_type, "train"
            )
            create_dataset_classify_yolo_pred(
                imgs, test_idxs, data_root, data_type, "test"
            )

    if args.train:
        # train YOLO
        yolo_result_dir = train_classify(data_name, data_type)
    else:
        # only prediction
        v_num = args.version
        yolo_result_dir = f"runs/{data_name}/{data_type}"
        if v_num is not None:
            yolo_result_dir += f"-v{v_num}"

    # prediction
    train_paths = sorted(
        glob(os.path.join(f"datasets/{data_name}", data_type, "train", "**", "*.jpg"))
    )
    test_paths = sorted(
        glob(os.path.join(f"datasets/{data_name}", data_type, "test", "**", "*.jpg"))
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
