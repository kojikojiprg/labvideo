import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm


def create_classification_annotation_dataset(data, idxs, dataset_dir, data_type, stage):
    for idx in idxs:
        aid, label, img_name = data[idx]

        if data_type == "label":
            lbl_txt = label  # A11~C42
        elif data_type == "label_type":
            lbl_txt = label[0]  # only A, B, C
        else:
            raise ValueError

        img_path = os.path.join(dataset_dir, stage, lbl_txt, img_name)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        shutil.copyfile(os.path.join("annotation/images", img_name), img_path)


def create_classification_dataset(data, idxs, dataset_dir, data_type, stage):
    label_counts = {}
    video_counts_per_label = {}
    video_names = []
    for i, idx in enumerate(tqdm(idxs, ncols=100)):
        key, label, img_path = data[idx]
        img = cv2.imread(img_path)
        if len(img.shape) != 3:
            raise ValueError(f"{img.shape}")

        if data_type == "label":
            if "_" in label:
                label = label.split("_")[0]  # delete surfix
            lbl_txt = label  # A11~C42
        elif data_type == "label_type":
            lbl_txt = label[0]  # only A, B, C
        else:
            raise ValueError

        img_path = os.path.join(dataset_dir, stage, lbl_txt, f"{i:04d}.jpg")
        img_dir = os.path.dirname(img_path)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir, exist_ok=False)

        cv2.imwrite(img_path, img)

        if lbl_txt not in label_counts:
            label_counts[lbl_txt] = 0
            video_counts_per_label[lbl_txt] = []
        label_counts[lbl_txt] += 1
        video_name = key.split("-")[0]
        if video_name not in video_counts_per_label[lbl_txt]:
            video_counts_per_label[lbl_txt].append(video_name)
        if video_name not in video_names:
            video_names.append(video_name)

    label_counts = list(label_counts.items())
    label_counts = sorted(label_counts, key=lambda x: x[0])

    dataset_summary = []
    for label, count in label_counts:
        video_count = len(video_counts_per_label[label])
        dataset_summary.append((label, count, video_count))

    assert sum([d[1] for d in dataset_summary]) == len(idxs)
    dataset_summary.append(("total", len(idxs), len(video_names)))
    path = f"{dataset_dir}/summary_{stage}.tsv"
    np.savetxt(path, dataset_summary, "%s", delimiter="\t")


def create_anomaly_detection_dataset(data, idxs, dataset_dir, stage):
    data = np.array(data, dtype=str)
    data = data[idxs]
    path = f"{dataset_dir}/{stage}.tsv"
    np.savetxt(path, data, delimiter="\t", fmt="%s")

    dataset_summary = {0: 0, 1: 0}
    for d in tqdm(data, ncols=100):
        label = int(d[1])
        dataset_summary[label] += 1
    dataset_summary = list(dataset_summary.items())
    dataset_summary = sorted(dataset_summary, key=lambda x: x[0])
    assert sum([d[1] for d in dataset_summary]) == len(idxs)
    dataset_summary.append(("total", len(idxs)))
    path = f"{dataset_dir}/summary_{stage}.tsv"
    np.savetxt(path, dataset_summary, "%s", delimiter="\t")


def create_yolov8_finetuning_dataset(
    frame_paths, output_paths, video_ids, data_root, stage
):
    for frame_path, output_path in zip(frame_paths, output_paths):
        file_name = os.path.basename(frame_path).replace(".jpg", "")
        video_id = file_name.split("_")[0]
        if video_id not in video_ids:
            continue

        copy_frame_path = os.path.join(data_root, "images", stage, f"{file_name}.jpg")
        copy_output_path = os.path.join(data_root, "labels", stage, f"{file_name}.txt")

        img_dir = os.path.dirname(copy_frame_path)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        lbl_dir = os.path.dirname(copy_output_path)
        if not os.path.exists(lbl_dir):
            os.makedirs(lbl_dir)

        shutil.copyfile(frame_path, copy_frame_path)
        shutil.copyfile(output_path, copy_output_path)
