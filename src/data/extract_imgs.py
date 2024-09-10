import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

from src.utils import video


def collect_annotation_paint_images(ann_json, info_json):
    image_paths = sorted(glob(os.path.join("annotation/images", "*.jpg")))
    data = []
    for image_path in tqdm(image_paths, ncols=100):
        video_id, aid, _ = os.path.basename(image_path).split("_")
        if video_id not in info_json:
            # tqdm.write(f"{video_id} is not in info.json")
            continue

        ann_lst = [ann for ann in ann_json[video_id] if ann["reply"] == aid]
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

        data.append((aid, label, os.path.basename(image_path)))

    return data


def extract_images_classify_dataset(video_name, ann_json, th_sec, th_iou, is_finetuned):
    if is_finetuned:
        str_finetuned = "_finetuned"
    else:
        str_finetuned = ""

    # get data
    ann_lst = np.loadtxt(
        os.path.join(f"out/{video_name}/{video_name}_ann.tsv"),
        str,
        delimiter="\t",
        skiprows=1,
    )
    if len(ann_lst) == 0:
        return []
    yolo_preds = np.loadtxt(
        os.path.join(f"out/{video_name}/{video_name}_det{str_finetuned}.tsv"),
        str,
        delimiter="\t",
        skiprows=1,
    )
    cap = video.Capture(f"video/{video_name}.mp4")
    th_n_frame = np.ceil(cap.fps * th_sec).astype(int)

    ann_n_frames = [
        np.ceil(float(ann["time"]) * cap.fps).astype(int) for ann in ann_json
    ]

    data = []
    for n_frame in tqdm(ann_n_frames, ncols=100, desc=video_name):
        ret, frame = cap.read(n_frame)
        if not ret:
            print(f"frame not loaded from n_frame {n_frame} in {video_name}.mp4")
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # obtain yolo preds within th_n_frame
        th_min = max(n_frame - th_n_frame, 0)
        th_max = min(n_frame + th_n_frame, cap.frame_count)
        yp_n_frame = yolo_preds.T[0].astype(int)
        mask = (th_min <= yp_n_frame) & (yp_n_frame <= th_max)
        yolo_preds_tmp = yolo_preds[mask]
        if len(yolo_preds_tmp) == 0:
            continue

        ann_lst_tmp = ann_lst[ann_lst.T[0].astype(int) == n_frame]
        for ann in ann_lst_tmp:
            paint_bbox = ann[1:5].astype(np.float32)

            # extract yolo preds greater than th_iou
            ious = calc_ious(paint_bbox, yolo_preds_tmp[:, 1:5].astype(np.float32))
            yolo_preds_high_iou = yolo_preds_tmp[ious >= th_iou]

            label = ann[8]
            try:
                label = label.split("(")[1].replace(")", "")  # extract within bracket
            except IndexError:
                print("error label", video_name, label)
                continue

            key = f"{video_name}-{label}"
            for pred in yolo_preds_high_iou:
                x1, y1, x2, y2 = pred[1:5].astype(float).astype(int)
                data.append((key, label, frame[y1:y2, x1:x2]))

    del cap
    return data


def extract_images_anomaly_dataset(
    video_name, ann_json, data_root, th_sec, th_iou, is_finetuned
):
    if os.path.exists(f"{data_root}/images/{video_name}"):
        return _collect_anomaly_dataset(video_name, data_root)

    if is_finetuned:
        str_finetuned = "_finetuned"
    else:
        str_finetuned = ""

    # load annotation
    ann_lst = np.loadtxt(
        os.path.join(f"out/{video_name}/{video_name}_ann.tsv"),
        str,
        delimiter="\t",
        skiprows=1,
    )
    if len(ann_lst) == 0:
        return []

    # load yolo detection results
    yolo_preds = np.loadtxt(
        os.path.join(f"out/{video_name}/{video_name}_det{str_finetuned}.tsv"),
        str,
        delimiter="\t",
        skiprows=1,
    )

    # load video
    cap = video.Capture(f"video/{video_name}.mp4")
    th_n_frame = np.ceil(cap.fps * th_sec).astype(int)

    ann_n_frames = [
        np.ceil(float(ann["time"]) * cap.fps).astype(int) for ann in ann_json
    ]
    for n_frame in tqdm(ann_n_frames, ncols=100, desc=video_name):
        ret, frame = cap.read(n_frame)
        if not ret:
            print(f"frame not loaded from n_frame {n_frame} in {video_name}.mp4")
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # obtain yolo preds within th_n_frame
        th_min = max(n_frame - th_n_frame, 0)
        th_max = min(n_frame + th_n_frame, cap.frame_count)
        yp_n_frame = yolo_preds.T[0].astype(int)
        mask = (th_min <= yp_n_frame) & (yp_n_frame <= th_max)
        yolo_preds_tmp = yolo_preds[mask]
        if len(yolo_preds_tmp) == 0:
            continue

        ann_lst_tmp = ann_lst[ann_lst.T[0].astype(int) == n_frame]
        for ann in ann_lst_tmp:
            aid = ann[0]
            paint_bbox = ann[1:5].astype(np.float32)

            # extract yolo preds greater than th_iou
            ious = calc_ious(paint_bbox, yolo_preds_tmp[:, 1:5].astype(np.float32))
            yolo_preds_high_iou = yolo_preds_tmp[ious >= th_iou]
            yolo_preds_low_iou = yolo_preds_tmp[ious < th_iou]

            img_dir = f"{data_root}/images/{video_name}/1/"
            os.makedirs(img_dir, exist_ok=True)
            for i, pred in enumerate(yolo_preds_high_iou):
                # anomaly data
                x1, y1, x2, y2 = pred[1:5].astype(float).astype(int)
                img = frame[y1:y2, x1:x2]
                img_path = f"{img_dir}/{video_name}_{aid}_{i}.jpg"
                cv2.imwrite(img_path, img)

            img_dir = f"{data_root}/images/{video_name}/0/"
            os.makedirs(img_dir, exist_ok=True)
            for i, pred in enumerate(yolo_preds_low_iou):
                # normal data
                x1, y1, x2, y2 = pred[1:5].astype(float).astype(int)
                img = frame[y1:y2, x1:x2]
                img_path = f"{img_dir}/{video_name}_{aid}_{i}.jpg"
                cv2.imwrite(img_path, img)

    return _collect_anomaly_dataset(video_name, data_root)


def _collect_anomaly_dataset(video_name, data_root):
    # load image paths
    data = []
    anomaly_img_paths = sorted(glob(f"{data_root}/images/{video_name}/1/*.jpg"))
    for img_path in anomaly_img_paths:
        key = f"{video_name}-1"
        data.append((key, 1, img_path))
    normal_img_paths = sorted(glob(f"{data_root}/images/{video_name}/0/*.jpg"))
    for img_path in normal_img_paths:
        key = f"{video_name}-0"
        data.append((key, 0, img_path))

    return data


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
