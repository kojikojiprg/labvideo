import argparse
import os
import shutil
import sys
from glob import glob

import numpy as np
from tqdm import tqdm

sys.path.append(".")
from src.data import (
    collect_images_classification_dataset,
    create_classification_dataset,
    split_train_test_by_annotation,
    split_train_test_by_video,
)
from src.model.classify import pred_classify, train_classify
from src.utils import json_handler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_type",
        type=str,
        choices=["label", "label_type"],
        help="'label' or 'label_type'",
    )
    parser.add_argument(
        "split_type", type=str, choices=["all", "video"], help="'all' or 'video'"
    )

    # optional
    parser.add_argument("-iou", "--th_iou", required=False, type=float, default=0.1)
    parser.add_argument("-sec", "--th_sec", required=False, type=float, default=0.5)
    parser.add_argument(
        "-br", "--bbox_ratio", required=False, type=float, default=0.125
    )
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
    bbox_ratio = args.bbox_ratio

    data_name = f"sec{th_sec}_iou{th_iou}_br{bbox_ratio}"
    dataset_dir = f"datasets/classify/{data_name}/{split_type}/{data_type}"
    yolo_result_dir = f"runs/classify/{data_name}/{split_type}/{data_type}"

    # create dataset
    if args.create_dataset:
        if os.path.exists(dataset_dir):
            print("removing dataset at", dataset_dir)
            shutil.rmtree(dataset_dir)
        os.makedirs(dataset_dir, exist_ok=False)

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

        img_dir = "datasets/images"
        data = []
        for video_id, ann_lst in tqdm(ann_json.items(), ncols=100):
            if video_id not in info_json:
                tqdm.write(f"{video_id} is not in info.json")
                continue

            video_name = video_id_to_name[video_id]
            data += collect_images_classification_dataset(
                video_name, th_sec, th_iou, bbox_ratio, img_dir
            )

        np.random.seed(42)
        if split_type == "all":
            random_idxs = np.random.choice(
                np.arange(len(data)), len(data), replace=False
            )
            train_length = int(len(data) * 0.7)
            train_idxs = random_idxs[:train_length]
            test_idxs = random_idxs[train_length:]
        elif split_type == "video":
            train_idxs, test_idxs = split_train_test_by_video(data, video_id_to_name)
        # elif split_type == "annotation":
        #     train_idxs, test_idxs, removed_labels = split_train_test_by_annotation(data)
        #     path = f"{dataset_dir}/removed_labels.tsv"
        #     np.savetxt(path, np.array(removed_labels), "%s", delimiter="\t")
        #     print("saved", path)
        else:
            raise ValueError(
                f"{split_type} is not selected from 'all', 'video' or 'annotation'."
            )

        create_classification_dataset(data, train_idxs, dataset_dir, data_type, "train")
        create_classification_dataset(data, test_idxs, dataset_dir, data_type, "test")

    if args.train:
        # train YOLO
        yolo_result_dir = train_classify(data_name, data_type, split_type)
    else:
        # only prediction
        v_num = args.version
        if v_num is not None:
            yolo_result_dir += f"-v{v_num}"

    # prediction
    train_paths = glob(os.path.join(dataset_dir, "train", "**", "*.jpg"))
    test_paths = glob(os.path.join(dataset_dir, "test", "**", "*.jpg"))

    train_labels = np.loadtxt(f"{dataset_dir}/summary_train.tsv", str, delimiter="\t")[
        :-1
    ].T[0]
    test_labels = np.loadtxt(f"{dataset_dir}/summary_test.tsv", str, delimiter="\t")[
        :-1
    ].T[0]
    labels = train_labels.tolist() + test_labels.tolist()
    labels = np.unique(sorted(labels))

    results_train, missed_img_path_train = pred_classify(
        train_paths, "train", yolo_result_dir, labels
    )
    results_test, missed_img_path_test = pred_classify(
        test_paths, "test", yolo_result_dir, labels
    )

    # missed_imgs_dir = os.path.join(yolo_result_dir, "missed_images_test")
    # if os.path.exists(missed_imgs_dir):
    #     shutil.rmtree(missed_imgs_dir)
    # os.makedirs(missed_imgs_dir, exist_ok=True)
    # for path, label, pred_label in missed_img_path_test:
    #     img_name = os.path.basename(path)
    #     img_name = f"true-{label}_pred-{pred_label}_" + img_name
    #     move_path = os.path.join(missed_imgs_dir, img_name)
    #     shutil.copyfile(path, move_path)

    print("complete")
