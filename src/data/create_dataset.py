import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm


def create_classify_paint_dataset(data, idxs, data_root, data_type, stage):
    for idx in idxs:
        aid, label, img_name = data[idx]

        if data_type == "label":
            lbl_txt = label  # A11~C42
        elif data_type == "label_type":
            lbl_txt = label[0]  # only A, B, C
        else:
            raise ValueError

        img_path = os.path.join(data_root, data_type, stage, lbl_txt, img_name)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        shutil.copyfile(os.path.join("annotation/images", img_name), img_path)


def create_classify_dataset(data, idxs, data_root, data_type, stage):
    for i, idx in enumerate(tqdm(idxs, ncols=100)):
        key, label, img = data[idx]
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

        img_path = os.path.join(data_root, data_type, stage, lbl_txt, f"{i:04d}.jpg")
        img_dir = os.path.dirname(img_path)
        if not os.path.exists(img_dir):
            os.makedirs(os.path.dirname(img_path), exist_ok=False)

        cv2.imwrite(img_path, img)


def create_anomaly_dataset(data, idxs, data_root, stage):
    data = np.array(data, dtype=str)
    data = data[idxs]
    path = f"{data_root}/{stage}.tsv"
    np.savetxt(path, data, delimiter="\t", fmt="%s")


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
