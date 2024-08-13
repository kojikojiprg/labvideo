import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

from src.utils import video


def collect_paint_imgs(ann_json, info_json):
    image_paths = sorted(glob(os.path.join("annotation/images", "*.jpg")))
    data = []
    for image_path in image_paths:
        video_id, aid, _ = os.path.basename(image_path).split("_")
        if video_id not in info_json:
            tqdm.write(f"{video_id} is not in info.json")
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


# TODO: delete anomaly option after creating anomaly detection model
def extract_yolo_preds(video_name, th_sec, th_iou, data_type, is_finetuned):
    if is_finetuned:
        str_finetuned = "_finetuned"
    else:
        str_finetuned = ""

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
        os.path.join(f"out/{video_name}/{video_name}_det{str_finetuned}.tsv"),
        str,
        delimiter="\t",
        skiprows=1,
    )
    cap = video.Capture(f"video/{video_name}.mp4")
    th_n_frame = np.ceil(cap.fps * th_sec).astype(int)

    data = []
    for n_frame in tqdm(range(cap.frame_count), ncols=100, desc=video_name):
        frame = cap.read()[1]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # obtain yolo preds within th_n_frame
        th_min = max(n_frame - th_n_frame, 0)
        th_max = min(n_frame + th_n_frame, cap.frame_count)
        yp_n_frame = yolo_preds.T[0].astype(int)
        mask = (th_min <= yp_n_frame) & (yp_n_frame <= th_max)
        yolo_preds_tmp = yolo_preds[mask]
        if len(yolo_preds_tmp) == 0:
            continue

        ann_lst = annotation_lst[annotation_lst.T[0].astype(int) == n_frame]
        if len(ann_lst) == 0 and data_type == "anomaly":
            # append normal data
            key = f"{video_name}-0"
            for pred in yolo_preds_tmp:
                x1, y1, x2, y2 = pred[1:5].astype(float).astype(int)
                data.append((key, 0, frame[y1:y2, x1:x2]))
            continue

        for ann in ann_lst:
            paint_bbox = ann[1:5].astype(np.float32)

            # extract yolo preds greater than th_iou
            ious = calc_ious(paint_bbox, yolo_preds_tmp[:, 1:5].astype(np.float32))
            yolo_preds_anomaly = yolo_preds_tmp[ious >= th_iou]
            if data_type == "anomaly":
                yolo_preds_normal = yolo_preds_tmp[ious < th_iou]

            label = ann[8]
            try:
                label = label.split("(")[1].replace(")", "")  # extract within bracket
            except IndexError:
                print("error label", video_name, label)
                continue

            if data_type == "label" or data_type == "label_type":
                key = f"{video_name}-{label}"
                for pred in yolo_preds_anomaly:
                    x1, y1, x2, y2 = pred[1:5].astype(float).astype(int)
                    data.append((key, label, frame[y1:y2, x1:x2]))
            elif data_type == "anomaly":
                key = f"{video_name}-1"
                for pred in yolo_preds_anomaly:
                    x1, y1, x2, y2 = pred[1:5].astype(float).astype(int)
                    data.append((key, 1, frame[y1:y2, x1:x2]))
                key = f"{video_name}-0"
                for pred in yolo_preds_normal:
                    x1, y1, x2, y2 = pred[1:5].astype(float).astype(int)
                    data.append((key, 0, frame[y1:y2, x1:x2]))
            else:
                raise ValueError

    del cap
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
