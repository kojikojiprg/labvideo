import argparse
import os
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
        return None, []

    # except error annotation
    ann_lst = ann_lst[~np.all(ann_lst.T[1:5].astype(float) == 0.0, axis=0)]

    # extract frame number of each annotation
    n_frames = []
    for label in np.unique(ann_lst.T[8]):
        ann_lst_tmp = ann_lst[ann_lst.T[8] == label]
        n_frame = int(ann_lst_tmp[0, 0])
        n_frames.append(n_frame)

    return ann_lst, n_frames


def plot_bbox_anomaly_or_normal(video_name, th_sec, th_iou, bbox_ratio, is_finetuned):
    if is_finetuned:
        str_finetuned = "_finetuned"
    else:
        str_finetuned = ""

    # load annotation
    ann_lst, ann_n_frames = load_annotation(video_name)
    if ann_lst is None:
        return None

    # load yolo detection result
    yolo_preds = np.loadtxt(
        os.path.join(f"out/{video_name}/{video_name}_det{str_finetuned}.tsv"),
        str,
        delimiter="\t",
        skiprows=1,
    )

    # load video
    cap = video.Capture(f"video/{video_name}.mp4")
    frame_count = cap.frame_count
    frame_size = cap.size
    th_n_frame = np.ceil(cap.fps * th_sec).astype(int)

    # create writer
    wrt = video.Writer(
        f"out/{video_name}/{video_name}{str_finetuned}_iou{th_iou}_sec{th_sec}_br{bbox_ratio}.mp4",
        cap.fps,
        cap.size,
    )

    for n_frame in tqdm(range(frame_count), ncols=100, desc=video_name):
        ret, frame = cap.read()
        if not ret:
            print(f"frame not loaded from n_frame {n_frame} in {video_name}.mp4")
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        yolo_preds_tmp = yolo_preds[yolo_preds.T[0].astype(int) == n_frame]
        if len(yolo_preds_tmp) == 0:
            wrt.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            continue

        # frame contains abnormal labels?
        is_annotation_contained = False
        for ann_n_frame in ann_n_frames:
            th_min = max(ann_n_frame - th_n_frame, 0)
            th_max = min(ann_n_frame + th_n_frame, frame_count)
            if (th_min <= n_frame) and (n_frame <= th_max):
                is_annotation_contained = True
                break

        if not is_annotation_contained:
            # annotation is not found in this frame
            # plot normal data
            for pred in yolo_preds_tmp:
                x1, y1, x2, y2 = calc_resized_bbox(
                    pred[1:5].astype(float), bbox_ratio, frame_size
                )
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        else:
            ann_lst_tmp = ann_lst[ann_lst.T[0].astype(int) == ann_n_frame]

            # calc iou
            ious = []
            for ann in ann_lst_tmp:
                ann_bbox = ann[1:5].astype(np.float32)
                label = ann[8]

                # plot annotation
                x1, y1, x2, y2 = ann_bbox.astype(int)
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                frame = cv2.putText(
                    frame,
                    str(label),
                    (x1, y1),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (255, 255, 0),
                    1,
                )

                iou = calc_ious(ann_bbox, yolo_preds_tmp[:, 1:5].astype(np.float32))
                ious.append(iou.reshape(-1, 1))

            if len(ious) == 0:
                raise ValueError("not found annotation")
            elif len(ious) == 1:
                ious = ious[0]
            else:
                ious = np.concatenate(ious, axis=1)

            mask = np.any(ious >= th_iou, axis=1)
            yolo_preds_low_iou = yolo_preds_tmp[~mask]
            yolo_preds_high_iou = yolo_preds_tmp[mask]

            # plot normal data
            for pred in yolo_preds_low_iou:
                x1, y1, x2, y2 = calc_resized_bbox(
                    pred[1:5].astype(float), bbox_ratio, frame_size
                )
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # plot anomaly data
            for pred in yolo_preds_high_iou:
                x1, y1, x2, y2 = calc_resized_bbox(
                    pred[1:5].astype(float), bbox_ratio, frame_size
                )
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                frame = cv2.putText(
                    frame,
                    str(label),
                    (x1, y1),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (255, 0, 0),
                    1,
                )

        wrt.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    del cap, wrt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-iou", "--th_iou", required=False, type=float, default=0.1)
    parser.add_argument("-sec", "--th_sec", required=False, type=float, default=0.5)
    parser.add_argument(
        "-br", "--bbox_ratio", required=False, type=float, default=0.125
    )
    parser.add_argument("-f", "--finetuned_model", action="store_true")
    args = parser.parse_args()

    th_iou = args.th_iou
    th_sec = args.th_sec
    bbox_ratio = args.bbox_ratio

    if args.finetuned_model:
        str_finetuned = "_finetuned"
    else:
        str_finetuned = ""

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

    for video_id in tqdm(ann_json.keys(), ncols=100):
        if video_id not in info_json:
            tqdm.write(f"{video_id} is not in info.json")
            continue

        video_name = video_id_to_name[video_id]
        plot_bbox_anomaly_or_normal(
            video_name, th_sec, th_iou, bbox_ratio, args.finetuned_model
        )
    print("complete")
