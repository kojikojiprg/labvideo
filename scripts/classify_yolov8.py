import argparse
import os
import shutil
import sys
from glob import glob

import numpy as np
from tqdm import tqdm

sys.path.append(".")
from src.data import create_dataset_yolo_classify, extract_yolo_classify
from src.model.classify import pred_classify, train_classify
from src.utils import json_handler


def split_train_test_by_video(data, video_id_to_name):
    # data ("{video_name}-{label}, label, img")

    train_video_ids = np.loadtxt(
        "datasets/yolov8_finetuning/summary_dataset_train.tsv",
        dtype=str,
        delimiter="\t",
        skiprows=1,
    )[:-1, 0]
    test_video_ids = np.loadtxt(
        "datasets/yolov8_finetuning/summary_dataset_test.tsv",
        dtype=str,
        delimiter="\t",
        skiprows=1,
    )[:-1, 0]
    train_video_names = [video_id_to_name[_id] for _id in train_video_ids]
    test_video_names = [video_id_to_name[_id] for _id in test_video_ids]

    # split data per label
    train_idxs = []
    test_idxs = []
    for i, d in enumerate(data):
        video_name = d[0].split("-")[0]
        if video_name in train_video_names:
            train_idxs.append(i)
        elif video_name in test_video_names:
            test_idxs.append(i)

    return train_idxs, test_idxs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_type", type=str, help="'label' or 'label_type'")
    parser.add_argument(
        "split_type", type=str, help="'random' or 'video'"
    )

    # optional(dataset_type == 'yolo')
    parser.add_argument("-iou", "--th_iou", required=False, type=float, default=0.1)
    parser.add_argument("-sec", "--th_sec", required=False, type=float, default=1.0)

    # optional
    parser.add_argument("-f", "--finetuned_model", action="store_true")
    parser.add_argument(
        "-cd", "--create_dataset", required=False, action="store_true", default=False
    )
    parser.add_argument(
        "-tr", "--train", required=False, action="store_true", default=False
    )
    parser.add_argument("-v", "--version", required=False, type=int, default=None)
    args = parser.parse_args()

    data_type = args.data_type
    split_type = args.split_type
    th_iou = args.th_iou
    th_sec = args.th_sec

    if args.finetuned_model:
        str_finetuned = "_finetuned"
    else:
        str_finetuned = ""

    data_name = f"classify_yolo{str_finetuned}"
    data_name += f"_sec{th_sec}_iou{th_iou}{str_finetuned}"
    data_root = f"datasets/classify/{split_type}/{data_name}{str_finetuned}"
    os.makedirs(data_root, exist_ok=True)

    # create dataset
    if args.create_dataset:
        ann_json = json_handler.load("annotation/annotation.json")
        info_json = json_handler.load("annotation/info.json")
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

        data = []
        for video_id, ann_lst in tqdm(ann_json.items(), ncols=100):
            if video_id not in info_json:
                tqdm.write(f"{video_id} is not in info.json")
                continue

            video_name = video_id_to_name[video_id]
            data += extract_yolo_classify(
                video_name,
                ann_lst,
                th_sec,
                th_iou,
                is_finetuned=args.finetuned_model,
            )

        np.random.seed(42)
        if split_type == "random":
            random_idxs = np.random.choice(np.arange(len(data)), len(data))
            train_length = int(len(data) * 0.7)
            train_idxs = random_idxs[:train_length]
            test_idxs = random_idxs[train_length:]
        else:
            train_idxs, test_idxs = split_train_test_by_video(data, video_id_to_name)

        create_dataset_yolo_classify(data, train_idxs, data_root, data_type, "train")
        create_dataset_yolo_classify(data, test_idxs, data_root, data_type, "test")

    if args.train:
        # train YOLO
        yolo_result_dir = train_classify(data_name, data_type, split_type)
    else:
        # only prediction
        v_num = args.version
        yolo_result_dir = f"runs/split_type/{data_name}/{data_type}{str_finetuned}"
        if v_num is not None:
            yolo_result_dir += f"-v{v_num}"

    # prediction
    train_paths = glob(
        os.path.join(
            f"datasets/classify/{split_type}/{data_name}{str_finetuned}",
            data_type,
            "train",
            "**",
            "*.jpg",
        )
    )
    test_paths = glob(
        os.path.join(
            f"datasets/classify/{split_type}/{data_name}{str_finetuned}",
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

    missed_imgs_dir = os.path.join(
        yolo_result_dir, f"missed_images_test{str_finetuned}"
    )
    if os.path.exists(missed_imgs_dir):
        shutil.rmtree(missed_imgs_dir)
    os.makedirs(missed_imgs_dir, exist_ok=True)
    for path, label, pred_label in missed_img_path_test:
        img_name = os.path.basename(path)
        img_name = f"true-{label}_pred-{pred_label}_" + img_name
        move_path = os.path.join(missed_imgs_dir, img_name)
        shutil.copyfile(path, move_path)

    print("complete")
