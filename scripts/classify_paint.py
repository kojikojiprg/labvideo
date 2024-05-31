import argparse
import os
import shutil
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from tqdm import tqdm
from ultralytics import YOLO

sys.path.append(".")
from src.utils import json_handler

image_paths = sorted(glob(os.path.join("annotation/images", "*.jpg")))
annotation_json = json_handler.load("annotation/annotation.json")
info_json = json_handler.load("annotation/info.json")


parser = argparse.ArgumentParser()
parser.add_argument("data_type", type=str, help="'label' or 'label_type'")
parser.add_argument(
    "-cd", "--create_dataset", required=False, action="store_true", default=False
)
parser.add_argument(
    "-tr", "--train", required=False, action="store_true", default=False
)
args = parser.parse_args()

data_type = args.data_type
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

    create_dataset(ann_data, train_idxs, out_dir, data_type, "train")
    create_dataset(ann_data, test_idxs, out_dir, data_type, "test")


def model_pred(model, img_paths, stage, yolo_result_dir):
    results = []
    missed_img_paths = []
    for path in tqdm(img_paths):
        label = os.path.basename(os.path.dirname(path))
        pred = model(path)
        pred_label_id = pred[0].probs.top1
        names = pred[0].names
        pred_label = names[pred_label_id]
        results.append([label, pred_label])
        if label != pred_label:
            missed_img_paths.append([path, label, pred_label])

    results = np.array(results)
    cm = confusion_matrix(results.T[0], results.T[1], normalize="true")
    cmd = ConfusionMatrixDisplay(cm)
    cmd.plot(xticks_rotation="vertical", include_values=True, cmap="Blues")
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel("next state")
    plt.ylabel("pre state")
    plt.savefig(f"{yolo_result_dir}/cm_{stage}.jpg", bbox_inches="tight")
    plt.close()

    print(stage, accuracy_score(results.T[0], results.T[1]))
    return results, missed_img_paths


if args.train:
    # train YOLO
    model = YOLO("yolov8n-cls.pt")
    model.train(data=f"classify_paint/{data_type}/", epochs=100, task="classify")
else:
    yolo_result_dir = f"runs/classify/{data_type}"
    weights_path = os.path.join(yolo_result_dir, "weights", "last.pt")
    model = YOLO(weights_path)

    train_paths = sorted(
        glob(os.path.join("datasets/classify_paint", data_type, "train", "**", "*.jpg"))
    )
    test_paths = sorted(
        glob(os.path.join("datasets/classify_paint", data_type, "test", "**", "*.jpg"))
    )

    results_train, missed_img_path_train = model_pred(
        model, train_paths, "train", yolo_result_dir
    )
    results_test, missed_img_path_test = model_pred(
        model, test_paths, "test", yolo_result_dir
    )

    missed_imgs_dir = os.path.join(yolo_result_dir, "missed_images_test")
    shutil.rmtree(missed_imgs_dir)
    os.makedirs(missed_imgs_dir, exist_ok=True)
    for path, label, pred_label in missed_img_path_test:
        img_name = os.path.basename(path)
        img_name = f"true_{label}-pred_{pred_label}-" + img_name
        move_path = os.path.join(missed_imgs_dir, img_name)
        shutil.copyfile(path, move_path)
