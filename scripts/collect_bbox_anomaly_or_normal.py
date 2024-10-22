import argparse
import os
import shutil
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(".")
from src.data import calc_ious, calc_resized_bbox
from src.utils import json_handler, video


def load_annotation(video_name):
    ann_lst = np.loadtxt(
        os.path.join(f"out/{video_name}/{video_name}_ann.tsv"),
        str,
        delimiter="\t",
        skiprows=1,
    )
    if len(ann_lst) == 0:
        return [], []

    # except error annotation
    ann_lst = ann_lst[~np.all(ann_lst.T[1:5].astype(float) == 0.0, axis=0)]

    # extract frame number that contains annotations
    n_frames = np.unique(ann_lst.T[0]).astype(int)

    return ann_lst, n_frames


def is_annotation_contained_on(n_frame, ann_n_frames, th_n_frame, frame_count):
    if n_frame in ann_n_frames:
        return True, n_frame

    is_annotation_contained = False
    for ann_n_frame in ann_n_frames:
        th_min = max(ann_n_frame - th_n_frame, 0)
        th_max = min(ann_n_frame + th_n_frame, frame_count)
        if (th_min <= n_frame) and (n_frame <= th_max):
            is_annotation_contained = True
            break

    return is_annotation_contained, ann_n_frame


def crop_img(frame, bbox):
    x1, y1, x2, y2 = bbox.astype(int)
    h, w = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    return frame[y1:y2, x1:x2]


def save_bboxs(bboxs, frame, n_frame, img_dir, labels=None):
    for i, bbox in enumerate(bboxs):
        img = crop_img(frame.copy(), bbox)
        if labels is None:
            img_path = f"{img_dir}/{n_frame}_{i}.jpg"
        else:
            img_path = f"{img_dir}/{labels[i]}/{n_frame}_{i}.jpg"
            if not os.path.exists(os.path.dirname(img_path)):
                os.makedirs(os.path.dirname(img_path))
        cv2.imwrite(img_path, img)


def plot_normal_bboxs(bboxs, frame):
    for bbox in bboxs:
        x1, y1, x2, y2 = bbox.astype(int)
        green = (0, 255, 0)  # BGR
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), green, 1)

    return frame


def plot_anomaly_bboxs(labels, bboxs, frame):
    for bbox, label in zip(bboxs, labels):
        x1, y1, x2, y2 = bbox.astype(int)
        red = (0, 0, 255)  # BGR
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), red, 1)
        frame = cv2.putText(
            frame,
            str(label),
            (x1, y1),
            cv2.FONT_HERSHEY_COMPLEX,
            0.7,
            red,
            1,
        )

    return frame


def collect_bbox_anomaly_or_normal(video_name, th_sec, th_iou, bbox_ratio, img_dir):
    # load annotation
    ann_lst, ann_n_frames = load_annotation(video_name)
    if len(ann_lst) == 0:
        tqdm.write(f"{video_id} doesn't have correct annotation")
        return

    # load yolo detection result
    yolo_preds = np.loadtxt(
        os.path.join(f"out/{video_name}/{video_name}_det_finetuned.tsv"),
        str,
        delimiter="\t",
        skiprows=1,
    )

    # load video
    cap = video.Capture(f"video/{video_name}.mp4")
    frame_count = cap.frame_count
    th_n_frame = np.ceil(cap.fps * th_sec).astype(int)

    # create writer
    wrt = video.Writer(
        f"out/{video_name}/{video_name}_iou{th_iou}_sec{th_sec}_br{bbox_ratio}.mp4",
        cap.fps,
        cap.size,
    )

    # create save folder
    img_dir = f"{img_dir}/sec{th_sec}_iou{th_iou}_br{bbox_ratio}/{video_name}"
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    os.makedirs(img_dir)
    img_dir_normal = f"{img_dir}/0"
    os.makedirs(img_dir_normal)
    img_dir_anomaly = f"{img_dir}/1"
    os.makedirs(img_dir_anomaly)

    for n_frame in tqdm(range(frame_count), ncols=100, desc=video_name):
        ret, frame_raw = cap.read()
        frame = frame_raw.copy()
        if not ret:
            print(f"frame not loaded from n_frame {n_frame} in {video_name}.mp4")
            continue

        yolo_preds_tmp = yolo_preds[yolo_preds.T[0].astype(int) == n_frame]
        if len(yolo_preds_tmp) == 0:
            wrt.write(frame)
            continue

        # calc resized bboxs
        bboxs = [
            calc_resized_bbox(pred[1:5].astype(np.float32), bbox_ratio, cap.size)
            for pred in yolo_preds_tmp
        ]
        bboxs = np.array(bboxs)

        # frame contains abnormal labels?
        is_annotation_contained, ann_n_frame = is_annotation_contained_on(
            n_frame, ann_n_frames, th_n_frame, frame_count
        )

        if not is_annotation_contained:
            # annotation is not found in this frame
            frame = plot_normal_bboxs(bboxs, frame)
            save_bboxs(bboxs, frame_raw, n_frame, img_dir_normal)
        else:
            ann_lst_tmp = ann_lst[ann_lst.T[0].astype(int) == ann_n_frame]

            # calc iou
            labels = []
            ious = []
            for ann in ann_lst_tmp:
                ann_bbox = ann[1:5].astype(np.float32)
                label = ann[8]
                labels.append(label)

                # plot annotation
                x1, y1, x2, y2 = ann_bbox.astype(int)
                yellow = (0, 255, 255)  # BGR
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), yellow, 2)
                frame = cv2.putText(
                    frame,
                    str(label),
                    (x1, y1),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    yellow,
                    1,
                )

                iou = calc_ious(ann_bbox, bboxs)
                ious.append(iou)

            if len(ious) == 0:
                raise ValueError("not found annotation")
            elif len(ious) == 1:
                ious = ious[0].ravel()
                labels = np.array([labels[0] for _ in range(len(ious))])
            else:
                ious = np.array(ious)

                # select the label which has the highest iou score
                idxs = np.argmax(ious, axis=0).ravel()
                labels = np.array(labels)[idxs]
                ious = np.max(ious, axis=0)

            mask = ious >= th_iou
            bboxs_low_iou = bboxs[~mask]
            bboxs_high_iou = bboxs[mask]
            labels = labels[mask]

            frame = plot_normal_bboxs(bboxs_low_iou, frame)
            save_bboxs(bboxs_low_iou, frame_raw, n_frame, img_dir_normal)

            frame = plot_anomaly_bboxs(labels, bboxs_high_iou, frame)
            save_bboxs(bboxs_high_iou, frame_raw, n_frame, img_dir_anomaly, labels)

        wrt.write(frame)

    del cap, wrt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-iou", "--th_iou", required=False, type=float, default=0.1)
    parser.add_argument("-sec", "--th_sec", required=False, type=float, default=0.5)
    parser.add_argument(
        "-br", "--bbox_ratio", required=False, type=float, default=0.125
    )
    args = parser.parse_args()

    th_iou = args.th_iou
    th_sec = args.th_sec
    bbox_ratio = args.bbox_ratio

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
    for video_id in tqdm(list(ann_json.keys()), ncols=100):
        if video_id not in info_json:
            tqdm.write(f"{video_id} is not in info.json")
            continue

        video_name = video_id_to_name[video_id]
        collect_bbox_anomaly_or_normal(video_name, th_sec, th_iou, bbox_ratio, img_dir)
    print("complete")
