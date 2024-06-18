import argparse
import os
import shutil
import sys
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm
from ultralytics import YOLO

sys.path.append(".")
from src.utils import json_handler, video

image_paths = sorted(glob(os.path.join("annotation/images", "*.jpg")))
annotation_json = json_handler.load("annotation/annotation.json")
info_json = json_handler.load("annotation/info.json")


parser = argparse.ArgumentParser()
parser.add_argument("data_type", type=str, help="'label' or 'label_type'")
parser.add_argument("-iou", "--th_iou", required=False, type=float, default=0.1)
parser.add_argument("-sec", "--th_sec", required=False, type=float, default=1.0)
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
data_name = f"classify_yolo_sec{th_sec}_iou{th_iou}"
data_root = f"datasets/{data_name}"
os.makedirs(data_root, exist_ok=True)


def calc_ious(target_bbox, bboxs):
    bboxs = np.asarray(bboxs)
    a_area = (target_bbox[2] - target_bbox[0]) * (target_bbox[3] - target_bbox[1])
    b_area = (bboxs[:, 2] - bboxs[:, 0]) * (bboxs[:, 3] - bboxs[:, 1])

    intersection_xmin = np.maximum(target_bbox[0], bboxs[:, 0])
    intersection_ymin = np.maximum(target_bbox[1], bboxs[:, 1])
    intersection_xmax = np.minimum(target_bbox[2], bboxs[:, 2])
    intersection_ymax = np.minimum(target_bbox[3], bboxs[:, 3])

    intersection_w = np.maximum(0, intersection_xmax - intersection_xmin)
    intersection_h = np.maximum(0, intersection_ymax - intersection_ymin)

    intersection_area = intersection_w * intersection_h
    union_area = a_area + b_area - intersection_area

    return intersection_area / union_area


def extract_dataset_imgs(video_name, th_sec, th_iou):
    # get data
    annotation_lst = np.loadtxt(
        os.path.join(f"out/{video_name}/{video_name}_ann.tsv"),
        str,
        delimiter="\t",
        skiprows=1,
    )
    if len(annotation_lst) == 0:
        return []
    yolo_preds = np.loadtxt(
        os.path.join(f"out/{video_name}/{video_name}_det.tsv"),
        str,
        delimiter="\t",
        skiprows=1,
    )
    cap = video.Capture(f"video/{video_name}.mp4")
    th_n_frame = np.ceil(cap.fps * th_sec).astype(int)

    unique_labels = np.unique(annotation_lst.T[8])

    # get target annotation data
    target_ann_lst = []
    start_n_frames = []
    for label in unique_labels:
        ann = annotation_lst[annotation_lst.T[8] == label][0]
        target_ann_lst.append(ann)
        start_n_frames.append(int(ann[0]))
    target_ann_lst = np.array(target_ann_lst)
    start_n_frames = np.array(start_n_frames)

    imgs = []
    for n_frame in tqdm(range(cap.frame_count), ncols=100, desc=video_name):
        frame = cap.read()[1]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if n_frame not in start_n_frames:
            continue

        # obtain yolo preds within th_n_frame
        th_min = max(n_frame - th_n_frame, 0)
        th_max = min(n_frame + th_n_frame, cap.frame_count)
        mask = (th_min <= yolo_preds.T[0].astype(int)) & (
            yolo_preds.T[0].astype(int) <= th_max
        )
        yolo_preds_tmp = yolo_preds[mask]

        idxs = np.where(start_n_frames == n_frame)[0]
        for idx in idxs:
            ann = target_ann_lst[idx]
            paint_bbox = ann[1:5].astype(np.float32)

            # extract yolo preds greater than th_iou
            ious = calc_ious(paint_bbox, yolo_preds_tmp[:, 1:5].astype(np.float32))
            yolo_preds_tmp = yolo_preds_tmp[ious >= th_iou]
            # print(np.mean(ious), len(yolo_preds_tmp))

            label = ann[8]
            try:
                label = label.split("(")[1].replace(")", "")  # extract within bracket
                if "_" in label:
                    label = label.split("_")[0]  # delete surfix
            except IndexError:
                print("error label", video_name, label)
                continue

            for pred in yolo_preds_tmp:
                x1, y1, x2, y2 = pred[1:5].astype(float).astype(int)
                imgs.append((label, frame[y1:y2, x1:x2]))
    del cap
    return imgs


def create_dataset(imgs, idxs, data_root, data_type, stage):
    for i, idx in enumerate(tqdm(idxs)):
        label, img = imgs[idx]
        if len(img.shape) != 3:
            raise ValueError(f"{img.shape}")

        if data_type == "label":
            lbl_txt = label  # A11~C42
        elif data_type == "label_type":
            lbl_txt = label[0]  # only A, B, C
        else:
            raise ValueError

        img_path = os.path.join(data_root, data_type, stage, lbl_txt, f"{i:04d}.jpg")
        os.makedirs(os.path.dirname(img_path), exist_ok=True)

        cv2.imwrite(img_path, img)


if args.create_dataset:
    ann_json = json_handler.load("annotation/annotation.json")
    info_json = json_handler.load("annotation/info.json")
    paint_data_json = json_handler.load("annotation/paint_bbox.json")

    video_id_to_name = {
        data[0]: data[1].split(".")[0]
        for data in np.loadtxt(
            "annotation/annotation.tsv", str, delimiter="\t", skiprows=1, usecols=[1, 2]
        )
        if data[0] != "" and data[1] != ""
    }

    imgs = []
    for video_id, ann_lst in tqdm(ann_json.items(), ncols=100):
        if video_id not in info_json:
            tqdm.write(f"{video_id} is not in info.json")
            continue

        video_name = video_id_to_name[video_id]
        imgs += extract_dataset_imgs(video_name, th_sec, th_iou)

    np.random.seed(42)
    random_idxs = np.random.choice(np.arange(len(imgs)), len(imgs))

    train_length = int(len(imgs) * 0.7)
    train_idxs = random_idxs[:train_length]
    test_idxs = random_idxs[train_length:]

    create_dataset(imgs, train_idxs, data_root, data_type, "train")
    create_dataset(imgs, test_idxs, data_root, data_type, "test")


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
    cm = confusion_matrix(results.T[0], results.T[1])
    path = f"{yolo_result_dir}/cm_{stage}_num.jpg"
    cm_plot(cm, path)
    cm = confusion_matrix(results.T[0], results.T[1], normalize="true")
    path = f"{yolo_result_dir}/cm_{stage}_recall.jpg"
    cm_plot(cm, path)
    cm = confusion_matrix(results.T[0], results.T[1], normalize="pred")
    path = f"{yolo_result_dir}/cm_{stage}_precision.jpg"
    cm_plot(cm, path)
    cm = confusion_matrix(results.T[0], results.T[1], normalize="all")
    path = f"{yolo_result_dir}/cm_{stage}_f1.jpg"
    cm_plot(cm, path)

    path = f"{yolo_result_dir}/cm_{stage}_report.tsv"
    report = classification_report(
        results.T[0], results.T[1], digits=3, output_dict=True, zero_division=0
    )
    pd.DataFrame.from_dict(report).T.to_csv(path, sep="\t")
    return results, missed_img_paths


def cm_plot(cm, path):
    cmd = ConfusionMatrixDisplay(cm)
    cmd.plot(xticks_rotation="vertical", include_values=True, cmap="Blues")
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


v_num = args.version
yolo_result_dir = f"runs/{data_name}/{data_type}"
if args.train:
    # train YOLO
    model = YOLO("yolov8n-cls.pt")
    model.train(data=f"{data_name}/{data_type}/", epochs=100, task="classify")

    # get trained data dir
    dirs = sorted(glob("runs/classify/train*/"))
    trained_dir = dirs[-1]

    if os.path.exists(yolo_result_dir):
        dirs = sorted(glob(yolo_result_dir + "-v*/"))
        if len(dirs) == 0:
            v_num = 1
        else:
            last_dir = dirs[-1]
            v_num = int(os.path.dirname(last_dir).split("-")[-1].replace("v", "")) + 1
        yolo_result_dir = f"runs/{data_name}/{data_type}-v{v_num}"
        shutil.move(trained_dir, yolo_result_dir)
    else:
        shutil.move(trained_dir, yolo_result_dir)
    os.makedirs(yolo_result_dir, exist_ok=True)
else:
    # only prediction
    if v_num is not None:
        yolo_result_dir += f"-v{v_num}"

# prediction
weights_path = os.path.join(yolo_result_dir, "weights", "last.pt")
model = YOLO(weights_path)

train_paths = sorted(
    glob(os.path.join(f"datasets/{data_name}", data_type, "train", "**", "*.jpg"))
)
test_paths = sorted(
    glob(os.path.join(f"datasets/{data_name}", data_type, "test", "**", "*.jpg"))
)

results_train, missed_img_path_train = model_pred(
    model, train_paths, "train", yolo_result_dir
)
results_test, missed_img_path_test = model_pred(
    model, test_paths, "test", yolo_result_dir
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
