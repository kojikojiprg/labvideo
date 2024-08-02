import argparse
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
from ultralytics import YOLO

sys.path.append(".")
from src.data import create_dataset_yolov8_finetuning
from src.utils import yaml_handler


def summarize_dataset(output_paths, train_ids, test_ids, classes, data_root):
    label_counts = {c: 0 for c in classes}
    train_label_counts = {c: 0 for c in classes}
    train_label_counts_videos = {vid: {c: 0 for c in classes} for vid in train_ids}
    test_label_counts = {c: 0 for c in classes}
    test_label_counts_videos = {vid: {c: 0 for c in classes} for vid in test_ids}
    for output_path in output_paths:
        outputs = np.loadtxt(output_path, str, delimiter=" ")
        file_name = os.path.basename(output_path).replace(".txt", "")
        video_id = file_name.split("_")[0]
        for out in outputs:
            cls_id = int(out[0])
            label = classes[cls_id]

            label_counts[label] += 1
            if video_id in train_ids:
                train_label_counts[label] += 1
                train_label_counts_videos[video_id][label] += 1
            if video_id in test_ids:
                test_label_counts[label] += 1
                test_label_counts_videos[video_id][label] += 1

    df_train = pd.DataFrame(train_label_counts, index=["train"])
    df_train_videos = [
        pd.DataFrame(counts, index=[vid])
        for vid, counts in train_label_counts_videos.items()
    ]
    df_summary_train = pd.concat(df_train_videos + [df_train], axis=0)
    df_summary_train.to_csv(
        os.path.join(data_root, "summary_dataset_train.tsv"), sep="\t"
    )

    df_test = pd.DataFrame(test_label_counts, index=["test"])
    df_test_videos = [
        pd.DataFrame(counts, index=[vid])
        for vid, counts in test_label_counts_videos.items()
    ]
    df_summary_test = pd.concat(df_test_videos + [df_test], axis=0)
    df_summary_test.to_csv(
        os.path.join(data_root, "summary_dataset_test.tsv"), sep="\t"
    )

    df_total = pd.DataFrame(label_counts, index=["total"])
    df_summary = pd.concat([df_train, df_test, df_total], axis=0)
    df_summary.to_csv(os.path.join(data_root, "summary_dataset.tsv"), sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # optional
    parser.add_argument(
        "-cd", "--create_dataset", required=False, action="store_true", default=False
    )
    args = parser.parse_args()

    data_name = "yolov8_finetuning"
    data_root = f"datasets/{data_name}"
    os.makedirs(data_root, exist_ok=True)

    if args.create_dataset:
        classes = np.loadtxt(
            "annotation/yolov8_finetuning/frames/output/classes.txt",
            str,
            delimiter="\t",
        )
        frame_paths = sorted(glob("annotation/yolov8_finetuning/frames/*.jpg"))
        output_paths = []
        video_ids = []
        for frame_path in frame_paths:
            file_name = os.path.basename(frame_path).replace(".jpg", "")
            output_path = f"annotation/yolov8_finetuning/frames/output/{file_name}.txt"
            output_paths.append(output_path)
            video_ids.append(file_name.split("_")[0])
        video_ids = np.unique(video_ids)

        # split train and test
        train_ratio = 0.7
        np.random.seed(42)
        n_videos = len(video_ids)
        random_idx = np.random.choice(np.arange(n_videos), n_videos, replace=False)
        train_len = int(np.ceil(n_videos * train_ratio))
        train_idxs = random_idx[:train_len]
        test_idxs = random_idx[train_len:]
        train_video_ids = video_ids[train_idxs]
        test_video_ids = video_ids[test_idxs]

        create_dataset_yolov8_finetuning(
            frame_paths, output_paths, train_video_ids, data_root, "train"
        )
        create_dataset_yolov8_finetuning(
            frame_paths, output_paths, test_video_ids, data_root, "test"
        )

        names = {i: str(c) for i, c in enumerate(classes)}
        yaml = {
            "path": "yolov8_finetuning",
            "train": "images/train",
            "val": "images/test",
            "test": "images/test",
            "names": names,
        }
        yaml_path = os.path.join(data_root, f"{data_name}.yaml")
        yaml_handler.dump(yaml_path, yaml)

        summarize_dataset(
            output_paths, train_video_ids, test_video_ids, classes, data_root
        )

    yaml_path = os.path.join(data_root, "yolov8_finetuning.yaml")
    model = YOLO("models/yolo/yolov8n.pt")
    results = model.train(data=yaml_path, epochs=100, imgsz=640)
