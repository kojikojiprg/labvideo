import os
import shutil

import cv2
from tqdm import tqdm


def create_dataset_classify_paint(imgs, idxs, data_root, data_type, stage):
    for idx in idxs:
        label, img_name = imgs[idx]

        if data_type == "label":
            lbl_txt = label  # A11~C42
        elif data_type == "label_type":
            lbl_txt = label[0]  # only A, B, C
        else:
            raise ValueError

        img_path = os.path.join(data_root, data_type, stage, lbl_txt, img_name)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        shutil.copyfile(os.path.join("annotation/images", img_name), img_path)


def create_dataset_classify_yolo_pred(imgs, idxs, data_root, data_type, stage):
    for i, idx in enumerate(tqdm(idxs)):
        label, img = imgs[idx]
        if len(img.shape) != 3:
            raise ValueError(f"{img.shape}")

        if data_type == "label":
            if "_" in label:
                label = label.split("_")[0]  # delete surfix
            lbl_txt = label  # A11~C42
        elif data_type == "label_type":
            lbl_txt = label[0]  # only A, B, C
        elif data_type == "anomaly":
            lbl_txt = label[0]  # 0, 1
        else:
            raise ValueError

        img_path = os.path.join(data_root, data_type, stage, lbl_txt, f"{i:04d}.jpg")
        os.makedirs(os.path.dirname(img_path), exist_ok=True)

        cv2.imwrite(img_path, img)
